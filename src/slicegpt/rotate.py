# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter, SlicingConfig
from .model_utils import get_layer0_inputs, get_signals, get_encoder_layer0_inputs, get_decoder_layer0_inputs, get_cross_attn_ln_inputs
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors
from typing import List, Optional, Tuple

def _force_layer_on_device_inplace(layer: nn.Module, dtype: torch.dtype) -> None:
    layer.to(device=config.device, dtype=dtype)


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    # Only slice shortcut Q if it exists
    if hasattr(layer_adapter.layer, "attn_shortcut_Q") and layer_adapter.layer.attn_shortcut_Q is not None:
        layer_adapter.layer.attn_shortcut_Q = nn.Parameter(
            layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :]
        )



def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()
    
def _sanitize_Q(Q: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.isfinite(Q).all():
        print(f"[WARN] {name} had NaN/Inf -> replacing with identity")
        d0, d1 = Q.shape
        out = torch.zeros((d0, d1), dtype=Q.dtype, device=Q.device)
        k = min(d0, d1)
        out[:k, :k] = torch.eye(k, dtype=Q.dtype, device=Q.device)
        return out
    return Q


''''
def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]
'''       
        
def slice_embeddings(model_adapter, new_embedding_dimensions):
    embeddings = list(model_adapter.get_embeddings())
    if not embeddings:
        return

    # --- Case 1: list/tuple (original expected format) ---
    if isinstance(new_embedding_dimensions, (list, tuple)):
        dims = list(new_embedding_dimensions)

    # --- Case 2: dict-like ---
    elif isinstance(new_embedding_dimensions, dict):
        # Try to access keys 0..N-1 directly (works for defaultdict and regular dict)
        try:
            dims = [new_embedding_dimensions[i] for i in range(len(embeddings))]
        except (KeyError, TypeError):
            # Fallback: infer target dim from slicing_conf (works for T5 reloads)
            sc = getattr(model_adapter, "slicing_conf", None)
            inferred = None

            # Prefer a known post-slice dimension:
            if sc is not None:
                # attention_input_dimensions is typically {layer_idx: dim}
                aid = getattr(sc, "attention_input_dimensions", None)
                if isinstance(aid, dict) and len(aid) > 0:
                    inferred = next(iter(aid.values()))
                # or const_dimension if present
                if inferred is None:
                    inferred = getattr(sc, "const_dimension", None)

            if inferred is None:
                raise ValueError(
                    "Cannot infer embedding dimension (embedding_dimensions dict has no "
                    "0..N-1 keys and slicing_conf missing)."
                )

            # If single embedding (T5 shared), apply inferred dim to it.
            # If multiple embeddings, apply same inferred dim to all (reasonable default).
            dims = [int(inferred)] * len(embeddings)

    else:
        raise TypeError(f"Unsupported embedding dimension type: {type(new_embedding_dimensions)}")

    # Sanity check
    if len(dims) != len(embeddings):
        raise ValueError(f"Got {len(dims)} embedding dims for {len(embeddings)} embeddings: {dims}")

    for W, d in zip(embeddings, dims):
        W.weight.data = W.weight.data[:, :d]
        W.embedding_dim = d



def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension


def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    # Seq2seq models (e.g., T5/FLAN-T5) require slicing encoder+decoder with cross-attention.
    if getattr(model_adapter.config, "is_encoder_decoder", False) and hasattr(model_adapter, "get_encoder_layers"):
        rotate_and_slice_seq2seq(
            model_adapter,
            dataloader,
            slicing_scheduler,
            apply_mask=apply_mask,
            final_orientation=final_orientation,
        )
        return

    if model_adapter.parallel_blocks:
        logging.info("Using parallel rotate and slice")
        rotate_and_slice_parallel(
            model_adapter,
            dataloader,
            slicing_scheduler,
            apply_mask=apply_mask,
            final_orientation=final_orientation,
        )
    else:
        logging.info("Using sequential rotate and slice")
        rotate_and_slice_sequential(
            model_adapter,
            dataloader,
            slicing_scheduler,
            apply_mask=apply_mask,
            final_orientation=final_orientation,
        )



@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    logging.info("Rotate and slice sequential")
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # rotate and slice embeddings
    eig_val, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0]).to(device=Q.device, dtype=Q.dtype)
        Q = Q @ R
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        rotate_attention_inputs(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device, dtype=dtype), Q.to(device=config.device, dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)
        if final_orientation == 'random':
            
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0]).to(device=Q.device, dtype=Q.dtype)
            Q = Q @ R

        # CRITICAL FIX: Compute attention shortcut in float64 to avoid numerical errors
        # Keep in float64 until final assignment to preserve orthogonality
        attn_shortcut_full = torch.matmul(
            layer.attn_shortcut_Q.to(dtype=torch.float64),
            Q[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)]
        )
        layer.attn_shortcut_Q = nn.Parameter(attn_shortcut_full.to(dtype=dtype))
        rotate_attention_output(layer_adapter, Q)
        slice_attention_output(
            layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        )

        # Store Q1 (MLP input rotation) in float64 for accurate computation
        Q1 = Q.clone()

        layer.mlp_shortcut_Q = nn.Parameter(
            Q1.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        eig_val, Q2 = pca_calc(inps, ignore_masks)
        Q2 = Q2.to(device=config.device, dtype=torch.float64)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q2.shape[0], slicing_scheduler.get_embedding_dimensions()[0]).to(device=Q2.device, dtype=Q2.dtype)
            Q2 = Q2 @ R

        # CRITICAL FIX: Compute MLP shortcut in float64 to avoid numerical errors
        # The double matrix multiplication (Q1.T @ Q2) can accumulate errors in float16
        mlp_shortcut_full = torch.matmul(
            Q1.T[: slicing_scheduler.get_mlp_input_dimension(idx), :],
            Q2
        )
        layer.mlp_shortcut_Q = nn.Parameter(
            mlp_shortcut_full[:, : slicing_scheduler.get_mlp_output_dimension(idx)].to(dtype=dtype)
        )

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q2)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

        # Use Q2 for next layer
        Q = Q2

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This version works for models where the MLP block and the attention block are computed in parallel.
    """
    logging.info("Rotate and slice parallel blocks")
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    # rotate and slice embeddings
    _, Q = pca_calc(inps, ignore_masks)
    Q = Q.to(device=config.device)
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0]).to(device=Q.device, dtype=Q.dtype)
        Q = Q @ R
        
    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the inputs to match previous layer (both attention and mlp)
        rotate_attention_inputs(layer_adapter, Q)
        rotate_mlp_input(layer_adapter, Q)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # update the input signals to this layer, and re-run it
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device, dtype=dtype), Q.to(device=config.device, dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        # the simpler equivalent of get_signals
        outputs = []
        layer = layer.to(config.device)
        for layer_args_batch, layer_kwargs_batch in zip(args, kwargs):
            layer_args_batch, layer_kwargs_batch = map_tensors(
                [layer_args_batch, layer_kwargs_batch], device=config.device
            )
            out = layer(*layer_args_batch, **layer_kwargs_batch)
            if isinstance(out, tuple):
                out = out[layer_adapter.hidden_states_output_position]
            out = out.cpu()
            outputs.append(out)

        inps = outputs
        _, Q = pca_calc(inps, ignore_masks)

        if final_orientation == 'random':
            
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0]).to(device=Q.device, dtype=Q.dtype)
            Q = Q @ R

        # update shortcut matrix
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        rotate_attention_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

        # slice the shortcut (there is only one, we use attn_shortcut buffer)
        layer.attn_shortcut_Q = nn.Parameter(
            layer.attn_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)]
        )

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model_adapter: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = model_adapter.get_layers()

    # Get the input of the first layer norm and calculate the Q_1
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)

    # Rotate the embeddings.
    rotate_embeddings(model_adapter, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer_adapter, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))

        # Rotate the Attention output matrix
        rotate_attention_output(layer_adapter, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer_adapter, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))

        # Rotate MLP output
        rotate_mlp_output(layer_adapter, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model_adapter: ModelAdapter) -> None:
    """
    Slice embeddings / encoder / decoder / layernorm / lm_head to match model_adapter.slicing_conf.
    For T5: lm_head must be sliced to the same dim as shared and tied to it.
    """
    sc = model_adapter.slicing_conf
    if sc is None:
        raise ValueError("model_adapter.slicing_conf is None")

    hidden_size = int(model_adapter.hidden_size)

    # ---------- helpers ----------
    def _get(m: dict, i: int, default: int) -> int:
        if m is None:
            return int(default)
        # Try both string and int keys
        if str(i) in m:
            return int(m[str(i)])
        elif i in m:
            return int(m[i])
        else:
            return int(default)

    
    def _last(m: dict, default: int) -> int:
        if not isinstance(m, dict) or len(m) == 0:
            return int(default)

        # keys may be "0","1",..., or ints; may be sparse
        items = []
        for k, v in m.items():
            try:
                ki = int(k)
            except Exception:
                continue
            items.append((ki, v))

        if not items:
            return int(default)

        ki_max, v_max = max(items, key=lambda t: t[0])
        return int(v_max)


    # ---------- embedding dim (shared) ----------
    emb_dim = int(sc.const_dimension) if sc.const_dimension is not None else None
    if emb_dim is None:
        # For decoder-only models (e.g., Gemma) with per-layer dimensions,
        # embedding should match FIRST layer's input, not last layer's output
        if hasattr(sc, 'attention_input_dimensions') and sc.attention_input_dimensions:
            # Use first layer's dimension for embeddings
            emb_dim = int(next(iter(sc.attention_input_dimensions.values())))
            logging.info(f"Inferred embedding dimension from first layer input: {emb_dim}")
        else:
            raise ValueError("slicing_conf.const_dimension is None and cannot infer from per-layer dimensions")

    # 1) Slice embeddings to match first layer's expected input
    slice_embeddings(model_adapter, [emb_dim] * len(list(model_adapter.get_embeddings())))

    # IMPORTANT: encoder final dim can differ from emb_dim in your config
    # (e.g. mlp_output_dimensions["11"] == 768 in your screenshot)
    enc_final_dim = _last(getattr(sc, "mlp_output_dimensions", None), emb_dim)

    # 2) Slice ENCODER stack using per-layer maps
    enc_layers = model_adapter.get_encoder_layers() if hasattr(model_adapter, "get_encoder_layers") else model_adapter.get_layers()
    for i, la in enumerate(enc_layers):
        d_sa_in  = _get(getattr(sc, "attention_input_dimensions", None),  i, emb_dim)
        d_sa_out = _get(getattr(sc, "attention_output_dimensions", None), i, emb_dim)
        d_mlp_in = _get(getattr(sc, "mlp_input_dimensions", None),        i, emb_dim)
        d_mlp_out= _get(getattr(sc, "mlp_output_dimensions", None),       i, emb_dim)

        slice_attention_inputs(la, d_sa_in)
        slice_attention_output(la, d_sa_out)

        slice_mlp_input(la, d_mlp_in)
        slice_mlp_output(la, d_mlp_out)

    # 3) Slice DECODER stack using per-layer maps (self-attn + cross-attn + mlp)
    if hasattr(model_adapter, "get_decoder_layers"):
        dec_layers = model_adapter.get_decoder_layers()
        for j, la in enumerate(dec_layers):
            d_sa_in  = _get(getattr(sc, "attention_input_dimensions", None),  j, emb_dim)
            d_sa_out = _get(getattr(sc, "attention_output_dimensions", None), j, emb_dim)
            d_mlp_in = _get(getattr(sc, "mlp_input_dimensions", None),        j, emb_dim)
            d_mlp_out= _get(getattr(sc, "mlp_output_dimensions", None),       j, emb_dim)

            # decoder self-attn
            slice_attention_inputs(la, d_sa_in)
            slice_attention_output(la, d_sa_out)

            # decoder cross-attn: Q comes from decoder space, KV comes from encoder final space
            slice_cross_attention_inputs(la, new_q_dim=d_sa_out, new_kv_dim=enc_final_dim)
            # cross-attn output projects back to decoder stream; use d_mlp_in (matches your rotate code style)
            slice_cross_attention_output(la, d_mlp_in)

            # decoder MLP
            slice_mlp_input(la, d_mlp_in)
            slice_mlp_output(la, d_mlp_out)

    # 4) Slice pre-head layernorm and lm_head
    # For decoder-only models (Gemma): lm_head input should match final layer output
    # For encoder-decoder models (T5): lm_head is tied to shared embeddings

    # Determine lm_head dimension: use final layer's MLP output for decoder-only models
    if hasattr(model_adapter, "get_encoder_layers"):
        # T5-like: use emb_dim (tied to shared)
        lm_head_dim = emb_dim
    else:
        # Gemma-like: use final layer's output dimension
        lm_head_dim = enc_final_dim  # This was computed by _last() from mlp_output_dimensions

    pre_head_ln = model_adapter.get_pre_head_layernorm()
    if pre_head_ln is not None and hasattr(pre_head_ln, "weight") and pre_head_ln.weight is not None:
        pre_head_ln.weight.data = pre_head_ln.weight.data[:lm_head_dim].contiguous()
        if getattr(pre_head_ln, "bias", None) is not None and pre_head_ln.bias is not None:
            pre_head_ln.bias.data = pre_head_ln.bias.data[:lm_head_dim].contiguous()

    # 5) Slice lm_head
    if hasattr(model_adapter, "get_lm_head"):
        lm_head = model_adapter.get_lm_head()
        W = lm_head.weight.data[:, :lm_head_dim].contiguous()
        lm_head.weight = torch.nn.Parameter(W)
        lm_head.in_features = lm_head_dim

        # T5: tie to shared embeddings
        if hasattr(model_adapter, "get_encoder_layers"):
            shared = model_adapter.get_embeddings()[0]
            lm_head.weight = shared.weight  # tie

            # keep config consistent
            if hasattr(model_adapter.model, "config"):
                model_adapter.model.config.d_model = emb_dim
            if hasattr(model_adapter.model, "tie_weights"):
                try:
                    model_adapter.model.tie_weights()
                except Exception:
                    pass






def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)

'''
@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec
'''

@torch.no_grad()
def pca_calc(
    X: List[torch.Tensor],
    ignore_masks: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robust PCA direction extraction.

    Returns (eig_val, eigen_vec) where eigen_vec columns are principal directions.
    Decomposition is done on CPU for numerical robustness.
    """
    cleanup_memory()

    dev = config.device
    H = None
    d = None

    rows: List[torch.Tensor] = []

    for idx, X_batch in enumerate(X):
        Xb = X_batch.to(device=dev, dtype=torch.float64)

        # -----------------------------
        # NEW 1) sanitize activations
        # -----------------------------
        if not torch.isfinite(Xb).all():
            # replace NaN/Inf with 0 so covariance doesn't become NaN
            Xb = torch.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)

        d = Xb.shape[-1] if d is None else d

        if ignore_masks is not None:
            m = ignore_masks[idx].to(device=dev)
            Xb = Xb * m.unsqueeze(-1).to(dtype=Xb.dtype)

            flat = Xb.reshape(-1, d)
            flat_m = m.reshape(-1)
            if flat_m.numel() == flat.shape[0]:
                flat = flat[flat_m != 0]
            rows.append(flat)
        else:
            rows.append(Xb.reshape(-1, d))

        # covariance / gram accumulation
        H_batch = torch.sum(Xb.mT @ Xb, dim=0)

        # -----------------------------
        # NEW 2) sanitize H_batch
        # -----------------------------
        if not torch.isfinite(H_batch).all():
            H_batch = torch.nan_to_num(H_batch, nan=0.0, posinf=0.0, neginf=0.0)

        H = H_batch if H is None else (H + H_batch)

    assert H is not None and d is not None

    # -----------------------------
    # NEW 3) fail fast if H is bad
    # -----------------------------
    if not torch.isfinite(H).all():
        raise RuntimeError("PCA failed: H (covariance) contains NaN/Inf. Activations are exploding upstream.")

    # --- Regularize strongly (adaptive) ---
    diag_mean = torch.diagonal(H).mean().abs().clamp_min(1e-12)
    eye = torch.eye(d, device=dev, dtype=torch.float64)

    H_cpu = (H + (1e-2 * diag_mean) * eye).cpu()
    try:
        evals, evecs = torch.linalg.eigh(H_cpu)
        evals = evals.to(dev)
        evecs = evecs.to(dev)
    except Exception:
        H_cpu2 = (H + (1e-1 * diag_mean) * eye).cpu()
        try:
            evals, evecs = torch.linalg.eigh(H_cpu2)
            evals = evals.to(dev)
            evecs = evecs.to(dev)
        except Exception:
            Xall = torch.cat([r for r in rows if r.numel() > 0], dim=0)
            Xall_cpu = Xall.cpu()
            Xall_cpu = Xall_cpu - Xall_cpu.mean(dim=0, keepdim=True)

            try:
                Xall_cpu = Xall_cpu + 1e-6 * torch.randn_like(Xall_cpu)
                _, S, Vh = torch.linalg.svd(Xall_cpu, full_matrices=False)
                evals = (S ** 2).to(dev)
                evecs = Vh.T.to(dev)
            except Exception:
                try:
                    q = min(d, max(8, min(d, Xall_cpu.shape[0] // 2)))
                    U, S, V = torch.pca_lowrank(Xall_cpu, q=q, center=True)
                    Qfull, _ = torch.linalg.qr(
                        torch.cat([V, torch.randn(d, d - V.shape[1], dtype=V.dtype)], dim=1)
                    )
                    evecs = Qfull.to(dev)
                    evals = torch.cat([S**2, torch.zeros(d - S.numel(), dtype=S.dtype)], dim=0).to(dev)
                except Exception:
                    evals = torch.ones(d, device=dev, dtype=torch.float64)
                    evecs = torch.eye(d, device=dev, dtype=torch.float64)

    # Sort descending
    idx_sort = torch.argsort(evals, descending=True)
    eig_val = evals[idx_sort]
    eigen_vec = evecs[:, idx_sort]

    # -----------------------------
    # NEW 4) final sanity check
    # -----------------------------
    if not torch.isfinite(eig_val).all() or not torch.isfinite(eigen_vec).all():
        raise RuntimeError("PCA produced NaN/Inf eigenpairs. Upstream activations likely unstable (try float32 slicing).")

    return eig_val, eigen_vec



'''
def rotate_cross_attention_inputs(layer_adapter, Q_dec: torch.Tensor, Q_enc: torch.Tensor) -> None:
    """
    Rotate decoder cross-attention projections:
      - Q projection is fed by decoder hidden states -> rotate with Q_dec
      - K/V projections are fed by encoder hidden states -> rotate with Q_enc

    This assumes your decoder layer adapter exposes:
      - get_cross_attention_q_input() -> Linear
      - get_cross_attention_kv_inputs() -> list[Linear] (k, v)
    """
    # Q
    Wq = layer_adapter.get_cross_attention_q_input()
    dtype = Wq.weight.dtype
    Wq_ = Wq.weight.data.to(device=config.device, dtype=torch.float64)
    Wq.weight.data = torch.matmul(Wq_, Q_dec).to(device="cpu", dtype=dtype)

    # K, V
    for W in layer_adapter.get_cross_attention_kv_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q_enc).to(device="cpu", dtype=dtype)
'''
def rotate_cross_attention_inputs(layer_adapter, Q_dec: torch.Tensor) -> None:
    """
    Rotate decoder cross-attention projections:
      - Q projection is fed by decoder hidden states -> rotate with Q_dec
      - K/V projections are fed by encoder hidden states -> DO NOT rotate here.
        (Encoder hidden states you feed are already in their final basis; we only slice K/V later.)
    """
    # Rotate only Q
    Wq = layer_adapter.get_cross_attention_q_input()
    dtype = Wq.weight.dtype
    Wq_ = Wq.weight.data.to(device=config.device, dtype=torch.float64)
    Wq.weight.data = torch.matmul(Wq_, Q_dec).to(device="cpu", dtype=dtype)



def slice_cross_attention_inputs(layer_adapter, new_q_dim: int, new_kv_dim: int) -> None:
    Wq = layer_adapter.get_cross_attention_q_input()
    Wq.weight.data = Wq.weight.data[:, :new_q_dim]
    Wq.in_features = new_q_dim

    for W in layer_adapter.get_cross_attention_kv_inputs():
        W.weight.data = W.weight.data[:, :new_kv_dim]
        W.in_features = new_kv_dim

    # Slice residual shortcut to the *current hidden dim* (cross-attn residual add).
    # cross_attn_shortcut_Q must stay square: (new_q_dim x new_q_dim)
    layer = layer_adapter.layer
    if getattr(layer, "cross_attn_shortcut_Q", None) is not None:
        layer.cross_attn_shortcut_Q = nn.Parameter(layer.cross_attn_shortcut_Q[:new_q_dim, :new_q_dim])





def rotate_cross_attention_output(layer_adapter, Q_out: torch.Tensor) -> None:
    """
    Rotate decoder cross-attention output projection:
      o: [d_model, d_kv]  -> left-multiply by Q_out^T (and rotate bias if present)
    """
    Wo = layer_adapter.get_cross_attention_output()
    dtype = Wo.weight.data.dtype
    Wo_ = Wo.weight.data.to(device=config.device, dtype=torch.float64)
    Wo.weight.data = torch.matmul(Q_out.T, Wo_).to(device="cpu", dtype=dtype)

    if Wo.bias is not None:
        b = Wo.bias.data.to(device=config.device, dtype=torch.float64)
        Wo.bias.data = torch.matmul(Q_out.T, b).to(device="cpu", dtype=dtype)


def slice_cross_attention_output(layer_adapter, new_out_dim: int) -> None:
    """
    Slice decoder cross-attention output projection:
      o: [d_model, d_kv] -> slice rows to new_out_dim
    """
    Wo = layer_adapter.get_cross_attention_output()
    Wo.weight.data = Wo.weight.data[:new_out_dim, :]
    if Wo.bias is not None:
        Wo.bias.data = Wo.bias.data[:new_out_dim]
    Wo.out_features = new_out_dim


@torch.no_grad()
def rotate_and_slice_seq2seq(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader,
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = "pca",
) -> None:
    model_adapter.model.to(device=config.device, dtype=torch.float32)
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    # -------------------------
    # 0) Collect encoder layer0 inputs (after embeddings)
    # -------------------------
    enc_inps, enc_args, enc_kwargs, enc_ignore_masks = [], [], [], []
    dec_batches = []

    for batch in dataloader:
        dec_batches.append(batch)
        inp0, args0, kwargs0 = get_encoder_layer0_inputs(model_adapter, batch)
        enc_inps.append(inp0)     # (B,T,768) initially
        enc_args.append(args0)
        enc_kwargs.append(kwargs0)
        if apply_mask:
            enc_ignore_masks.append(batch["attention_mask"])

    enc_layers = model_adapter.get_encoder_layers()
    dec_layers = model_adapter.get_decoder_layers()

    # -------------------------
    # schedulers (encoder + decoder separate)
    # -------------------------
    enc_sched: SlicingScheduler = copy.deepcopy(slicing_scheduler)
    dec_sched: SlicingScheduler = copy.deepcopy(slicing_scheduler)
    enc_sched.setup(hidden_size=model_adapter.hidden_size, layers_num=len(enc_layers), parallel_blocks=False)
    dec_sched.setup(hidden_size=model_adapter.hidden_size, layers_num=len(dec_layers), parallel_blocks=False)

    # -------------------------
    # 1) Rotate embeddings (768x768), THEN slice to emb_dim
    # -------------------------
    _, Q0 = pca_calc(enc_inps, enc_ignore_masks)   # (768,768)
    Q0 = Q0.to(device=config.device, dtype=torch.float64)

    emb_dim = int(enc_sched.get_embedding_dimensions()[0])  # e.g. 688

    # rotate embeddings with FULL 768x768
    rotate_embeddings(model_adapter, Q0)
    # slice embeddings to emb_dim
    slice_embeddings(model_adapter, [emb_dim])

    # IMPORTANT: also rotate+truncate cached encoder inputs to emb_dim
    enc_inps = [
        (inp.to(config.device, dtype=dtype) @ Q0.to(config.device, dtype=dtype))[:, :, :emb_dim].cpu()
        for inp in enc_inps
    ]

    # From now on we operate in emb_dim space, and the basis is already "baked in"
    Q_enc = torch.eye(emb_dim, device=config.device, dtype=torch.float64)

    # Tie lm_head to shared (T5 invariant)
    model = model_adapter.model
    try:
        if hasattr(model, "shared") and hasattr(model, "lm_head"):
            d = int(model.shared.weight.shape[1])  # should be emb_dim
            model.lm_head.weight.data = model.lm_head.weight.data[:, :d]
            model.lm_head.in_features = d
            model.lm_head.weight = model.shared.weight
    except Exception as e:
        logging.warning(f"Could not force-tie lm_head to shared: {e}")

    # keep decoder scheduler embedding config consistent
    _ = dec_sched.get_embedding_dimensions()

    # Replace the ENCODER loop in rotate_and_slice_seq2seq
    # This is a simplified version that avoids the complexity of get_signals()

    # Replace the ENCODER loop in rotate_and_slice_seq2seq
    # CRITICAL FIX: Slice BEFORE rotating to match dimensions

    # Replace the ENCODER loop in rotate_and_slice_seq2seq
    # CRITICAL FIX: Update enc_args to use rotated/sliced embeddings before layer 0

    # -------------------------
    # 2) Encoder stack (SIMPLIFIED WITH CORRECT ORDER)
    # -------------------------

    # Start with embedding dimension
    cur_dim = int(model_adapter.model.shared.weight.shape[1])  # 688 after embedding slice
    target_dim = cur_dim  # For ConstSlicingScheduler, all layers use same dim

    logging.info(f"Encoder target dimension: {target_dim}")

    # Identity rotation throughout (embeddings already rotated globally)
    Q_identity = torch.eye(target_dim, device=config.device, dtype=torch.float64)

    # CRITICAL FIX: Update enc_args to use the rotated/sliced enc_inps (688-dim)
    # Before layer 0 runs, enc_args still point to original 768-dim hidden states
    for i, inp in enumerate(enc_inps):
        enc_args[i] = model_adapter.get_encoder_layers()[0].get_updated_args(inp, enc_args[i])

    for idx, layer_adapter in enumerate(tqdm(enc_layers, desc="Rotating/slicing encoder", unit="layer")):
        layer = layer_adapter.layer
        
        # ============================================================
        # SELF-ATTENTION - SLICE THEN ROTATE
        # ============================================================
        
        # Set shortcut first
        layer.attn_shortcut_Q = nn.Parameter(Q_identity.to(dtype=dtype, device="cpu"))
        
        # CRITICAL: Slice FIRST to reduce weight matrix dimensions
        slice_attention_inputs(layer_adapter, target_dim)
        # THEN rotate in the sliced space
        rotate_attention_inputs(layer_adapter, Q_identity)
        
        # Same for output
        slice_attention_output(layer_adapter, target_dim)
        rotate_attention_output(layer_adapter, Q_identity)
        
        # ============================================================
        # MLP - SLICE THEN ROTATE
        # ============================================================
        
        # Set shortcut first
        layer.mlp_shortcut_Q = nn.Parameter(Q_identity.to(dtype=dtype, device="cpu"))
        
        # Slice then rotate
        slice_mlp_input(layer_adapter, target_dim)
        rotate_mlp_input(layer_adapter, Q_identity)
        
        slice_mlp_output(layer_adapter, target_dim)
        rotate_mlp_output(layer_adapter, Q_identity)
        
        # ============================================================
        # Run layer to get outputs for next layer
        # ============================================================
        
        _force_layer_on_device_inplace(layer, dtype)
        
        # Run layer to get outputs for next layer
        new_enc_inps = []
        for enc_arg, enc_kwarg in zip(enc_args, enc_kwargs):
            # Move to device
            enc_arg_device = tuple(
                arg.to(device=config.device, dtype=dtype) if isinstance(arg, torch.Tensor) else arg
                for arg in enc_arg
            )
            enc_kwarg_device = {
                k: v.to(device=config.device, dtype=dtype) if isinstance(v, torch.Tensor) else v
                for k, v in enc_kwarg.items()
            }
            
            # Forward pass
            out = layer(*enc_arg_device, **enc_kwarg_device)
            if isinstance(out, tuple):
                out = out[0]  # hidden_states
            new_enc_inps.append(out.cpu())
        
        # Update for next layer
        enc_inps = new_enc_inps
        for i, inp in enumerate(enc_inps):
            enc_args[i] = layer_adapter.get_updated_args(inp, enc_args[i])
        
        layer.to("cpu")
        cleanup_memory()
        
        # Record dimensions
        enc_sched.slicing_conf.attention_input_dimensions[str(idx)]  = int(target_dim)
        enc_sched.slicing_conf.attention_output_dimensions[str(idx)] = int(target_dim)
        enc_sched.slicing_conf.mlp_input_dimensions[str(idx)]        = int(target_dim)
        enc_sched.slicing_conf.mlp_output_dimensions[str(idx)]       = int(target_dim)

    # Encoder outputs
    enc_outs = enc_inps
    enc_kv_dim = enc_outs[0].shape[-1]

    logging.info(f"Encoder final output dimension: {enc_kv_dim}")


    # Replace the ENTIRE decoder section in rotate_and_slice_seq2seq
    # This version properly handles encoder_hidden_states throughout

    # -------------------------
    # 3) Decoder stack - FIXED to provide encoder_hidden_states
    # -------------------------

    # Prepare decoder layer 0 inputs
    dec_inps, dec_args, dec_kwargs, dec_ignore_masks = [], [], [], []
    for batch, enc_h in zip(dec_batches, enc_outs):
        inp0, args0, kwargs0 = get_decoder_layer0_inputs(model_adapter, batch, encoder_hidden_states=enc_h)
        dec_inps.append(inp0)
        dec_args.append(args0)
        dec_kwargs.append(kwargs0)
        if apply_mask:
            dec_ignore_masks.append(batch.get("decoder_attention_mask", batch["attention_mask"]))

    # For ConstSlicingScheduler, all layers use the same dimension
    target_dim = emb_dim  # Same as embedding dimension (688)

    logging.info(f"Decoder target dimension: {target_dim}")

    for j, layer_adapter in enumerate(tqdm(dec_layers, desc="Rotating/slicing decoder", unit="layer")):
        layer = layer_adapter.layer
        
        # Identity rotation (embeddings already rotated globally)
        Q_identity = torch.eye(target_dim, device=config.device, dtype=torch.float64)
        
        # ============================================================
        # 1. SELF-ATTENTION
        # ============================================================
        
        layer.attn_shortcut_Q = nn.Parameter(Q_identity.to(dtype=dtype, device="cpu"))
        
        slice_attention_inputs(layer_adapter, target_dim)
        rotate_attention_inputs(layer_adapter, Q_identity)
        
        slice_attention_output(layer_adapter, target_dim)
        rotate_attention_output(layer_adapter, Q_identity)
        
        # ============================================================
        # 2. CROSS-ATTENTION
        # ============================================================
        
        layer.cross_attn_shortcut_Q = nn.Parameter(Q_identity.to(dtype=dtype, device="cpu"))
        
        # Cross-attention Q: decoder hidden state (target_dim)
        ca_q = layer_adapter.get_cross_attention_q_input()
        if ca_q.weight.shape[1] != target_dim:
            ca_q.weight.data = ca_q.weight.data[:, :target_dim].contiguous()
            ca_q.in_features = target_dim
        
        dtype_q = ca_q.weight.dtype
        W_q = ca_q.weight.data.to(device=config.device, dtype=torch.float64)
        ca_q.weight.data = torch.matmul(W_q, Q_identity).to(device="cpu", dtype=dtype_q)
        
        # Cross-attention K, V: encoder output (enc_kv_dim)
        # NOT rotated (encoder already in final basis)
        for W in layer_adapter.get_cross_attention_kv_inputs():
            if W.weight.shape[1] != enc_kv_dim:
                W.weight.data = W.weight.data[:, :enc_kv_dim].contiguous()
                W.in_features = enc_kv_dim
        
        # Cross-attention output
        slice_cross_attention_output(layer_adapter, target_dim)
        rotate_cross_attention_output(layer_adapter, Q_identity)
        
        # ============================================================
        # 3. MLP
        # ============================================================
        
        layer.mlp_shortcut_Q = nn.Parameter(Q_identity.to(dtype=dtype, device="cpu"))
        
        slice_mlp_input(layer_adapter, target_dim)
        rotate_mlp_input(layer_adapter, Q_identity)
        
        slice_mlp_output(layer_adapter, target_dim)
        rotate_mlp_output(layer_adapter, Q_identity)
        
        # ============================================================
        # 4. Run layer to update hidden states for next layer
        # ============================================================
        
        _force_layer_on_device_inplace(layer, dtype)
        
        # CRITICAL FIX: Add encoder_hidden_states to ALL kwargs
        # This is what was missing - cross-attention needs encoder outputs!
        for i in range(len(dec_kwargs)):
            if 'encoder_hidden_states' not in dec_kwargs[i]:
                dec_kwargs[i]['encoder_hidden_states'] = enc_outs[i]
        
        # Now get_signals will work correctly because encoder_hidden_states is present
        new_dec_inps = []
        for dec_arg, dec_kwarg in zip(dec_args, dec_kwargs):
            dec_arg_device = tuple(
                arg.to(device=config.device, dtype=dtype) if isinstance(arg, torch.Tensor) else arg
                for arg in dec_arg
            )
            dec_kwarg_device = {
                k: v.to(device=config.device, dtype=dtype) if isinstance(v, torch.Tensor) else v
                for k, v in dec_kwarg.items()
            }
            
            # Forward pass with proper encoder states
            try:
                out = layer(*dec_arg_device, **dec_kwarg_device)
                if isinstance(out, tuple):
                    out = out[0]  # hidden_states
                new_dec_inps.append(out.cpu())
            except Exception as e:
                logging.error(f"Error in decoder layer {j} forward: {e}")
                raise
        
        dec_inps = new_dec_inps
        for i, inp in enumerate(dec_inps):
            dec_args[i] = layer_adapter.get_updated_args(inp, dec_args[i])
        
        layer.to("cpu")
        cleanup_memory()
        
        # Record dimensions
        dec_sched.slicing_conf.attention_input_dimensions[str(j)]  = int(target_dim)
        dec_sched.slicing_conf.attention_output_dimensions[str(j)] = int(target_dim)
        dec_sched.slicing_conf.mlp_input_dimensions[str(j)]        = int(target_dim)
        dec_sched.slicing_conf.mlp_output_dimensions[str(j)]       = int(target_dim)

    # -------------------------
    # 4) Finalize config and tie weights
    # -------------------------
    model_adapter.slicing_conf = dec_sched.slicing_conf.clone()
    model_adapter.slicing_conf.const_dimension = int(emb_dim)

    # Sanitize shortcut matrices
    for la in model_adapter.get_encoder_layers():
        lyr = la.layer
        if hasattr(lyr, "attn_shortcut_Q") and lyr.attn_shortcut_Q is not None:
            lyr.attn_shortcut_Q = nn.Parameter(_sanitize_Q(lyr.attn_shortcut_Q.data, "enc_attn").cpu())
        if hasattr(lyr, "mlp_shortcut_Q") and lyr.mlp_shortcut_Q is not None:
            lyr.mlp_shortcut_Q = nn.Parameter(_sanitize_Q(lyr.mlp_shortcut_Q.data, "enc_mlp").cpu())

    for la in model_adapter.get_decoder_layers():
        lyr = la.layer
        if getattr(lyr, "attn_shortcut_Q", None) is not None:
            lyr.attn_shortcut_Q = nn.Parameter(_sanitize_Q(lyr.attn_shortcut_Q.data, "dec_attn").cpu())
        if getattr(lyr, "cross_attn_shortcut_Q", None) is not None:
            lyr.cross_attn_shortcut_Q = nn.Parameter(_sanitize_Q(lyr.cross_attn_shortcut_Q.data, "dec_cross").cpu())
        if getattr(lyr, "mlp_shortcut_Q", None) is not None:
            lyr.mlp_shortcut_Q = nn.Parameter(_sanitize_Q(lyr.mlp_shortcut_Q.data, "dec_mlp").cpu())

    # CRITICAL: Tie lm_head to shared
    model = model_adapter.model
    if hasattr(model, "shared") and hasattr(model, "lm_head"):
        d = int(model.shared.weight.shape[1])
        if model.lm_head.weight.shape[1] != d:
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data[:, :d].contiguous())
            model.lm_head.in_features = d
        model.lm_head.weight = model.shared.weight
        logging.info(f"✓ Tied lm_head to shared embedding (dim={d})")


@torch.no_grad()
def rotate_and_slice_encoder_only(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader,
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = "pca",
) -> None:
    """
    Slice ONLY the encoder, leave decoder and embeddings untouched.
    This avoids the coordination problem while still reducing model size.
    """
    model_adapter.model.to(device=config.device, dtype=torch.float32)
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype
    
    logging.info("="*70)
    logging.info("ENCODER-ONLY SLICING (Decoder unchanged)")
    logging.info("="*70)
    
    # -------------------------
    # 1) Collect encoder inputs (NO embedding changes!)
    # -------------------------
    enc_inps, enc_args, enc_kwargs, enc_ignore_masks = [], [], [], []
    dec_batches = []
    
    for batch in dataloader:
        dec_batches.append(batch)
        inp0, args0, kwargs0 = get_encoder_layer0_inputs(model_adapter, batch)
        enc_inps.append(inp0)  # [B, T, 768]
        enc_args.append(args0)
        enc_kwargs.append(kwargs0)
        if apply_mask:
            enc_ignore_masks.append(batch["attention_mask"])
    
    enc_layers = model_adapter.get_encoder_layers()
    
    # -------------------------
    # 2) Setup scheduler for encoder only
    # -------------------------
    enc_sched = copy.deepcopy(slicing_scheduler)
    enc_sched.setup(
        hidden_size=model_adapter.hidden_size, 
        layers_num=len(enc_layers), 
        parallel_blocks=False
    )
    
    # Target dimension for encoder
    orig_dim = model_adapter.hidden_size  # 768
    target_dim = int(enc_sched.get_embedding_dimensions()[0])  # e.g., 688
    
    logging.info(f"Original dimension: {orig_dim}")
    logging.info(f"Encoder target dimension: {target_dim}")
    logging.info(f"Decoder dimension: {orig_dim} (UNCHANGED)")
    
    # -------------------------
    # 3) Compute rotation for encoder
    # -------------------------
    _, Q_enc = pca_calc(enc_inps, enc_ignore_masks)
    Q_enc = Q_enc.to(device=config.device, dtype=torch.float64)
    
    if final_orientation == 'random':
        R = random_orthogonal_upper_left(orig_dim, target_dim)
        R = R.to(device=Q_enc.device, dtype=Q_enc.dtype)
        Q_enc = Q_enc @ R
    
    # -------------------------
    # 4) Rotate and slice ENCODER layers only
    # -------------------------
    
    # Update encoder inputs to rotated space
    enc_inps = [
        (inp.to(config.device, dtype=dtype) @ Q_enc.to(config.device, dtype=dtype))[:, :, :target_dim].cpu()
        for inp in enc_inps
    ]
    
    # Update enc_args to use rotated inputs
    for i, inp in enumerate(enc_inps):
        enc_args[i] = enc_layers[0].get_updated_args(inp, enc_args[i])
    
    # Identity rotation in target space
    Q_identity = torch.eye(target_dim, device=config.device, dtype=torch.float64)
    
    for idx, layer_adapter in enumerate(tqdm(enc_layers, desc="Rotating/slicing encoder", unit="layer")):
        layer = layer_adapter.layer
        
        # Set shortcuts
        layer.attn_shortcut_Q = nn.Parameter(Q_identity.to(dtype=dtype, device="cpu"))
        layer.mlp_shortcut_Q = nn.Parameter(Q_identity.to(dtype=dtype, device="cpu"))
        
        # Slice then rotate attention
        slice_attention_inputs(layer_adapter, target_dim)
        rotate_attention_inputs(layer_adapter, Q_identity)
        slice_attention_output(layer_adapter, target_dim)
        rotate_attention_output(layer_adapter, Q_identity)
        
        # Slice then rotate MLP
        slice_mlp_input(layer_adapter, target_dim)
        rotate_mlp_input(layer_adapter, Q_identity)
        slice_mlp_output(layer_adapter, target_dim)
        rotate_mlp_output(layer_adapter, Q_identity)
        
        # Run layer
        _force_layer_on_device_inplace(layer, dtype)
        
        new_enc_inps = []
        for enc_arg, enc_kwarg in zip(enc_args, enc_kwargs):
            enc_arg_device = tuple(
                arg.to(device=config.device, dtype=dtype) if isinstance(arg, torch.Tensor) else arg
                for arg in enc_arg
            )
            enc_kwarg_device = {
                k: v.to(device=config.device, dtype=dtype) if isinstance(v, torch.Tensor) else v
                for k, v in enc_kwarg.items()
            }
            
            out = layer(*enc_arg_device, **enc_kwarg_device)
            if isinstance(out, tuple):
                out = out[0]
            new_enc_inps.append(out.cpu())
        
        enc_inps = new_enc_inps
        for i, inp in enumerate(enc_inps):
            enc_args[i] = layer_adapter.get_updated_args(inp, enc_args[i])
        
        layer.to("cpu")
        cleanup_memory()
    
    enc_outs = enc_inps
    enc_output_dim = enc_outs[0].shape[-1]
    
    logging.info(f"Encoder output dimension: {enc_output_dim}")
    
    # Slice encoder's final layer norm to match sliced dimension
    if hasattr(model_adapter.model.encoder, 'final_layer_norm'):
        ln = model_adapter.model.encoder.final_layer_norm
        
        # Handle RMSNorm (uses 'scale') vs LayerNorm (uses 'weight')
        if hasattr(ln, 'scale'):
            # RMSNorm
            if ln.scale.shape[0] != enc_output_dim:
                ln.scale.data = ln.scale.data[:enc_output_dim].contiguous()
                ln.normalized_shape = (enc_output_dim,)
                logging.info(f"  Sliced encoder RMSNorm to {enc_output_dim}")
        elif hasattr(ln, 'weight'):
            # Standard LayerNorm
            if ln.weight.shape[0] != enc_output_dim:
                ln.weight.data = ln.weight.data[:enc_output_dim].contiguous()
                if hasattr(ln, 'bias') and ln.bias is not None:
                    ln.bias.data = ln.bias.data[:enc_output_dim].contiguous()
                ln.normalized_shape = (enc_output_dim,)
                logging.info(f"  Sliced encoder LayerNorm to {enc_output_dim}")
    
    # -------------------------
    # 5) CRITICAL: Adjust decoder cross-attention K/V
    # Decoder stays 768-dim, but must accept 688-dim encoder outputs
    # -------------------------
    
    dec_layers = model_adapter.get_decoder_layers()
    
    logging.info(f"Adjusting decoder cross-attention for {enc_output_dim}-dim encoder outputs")
    
    for j, layer_adapter in enumerate(dec_layers):
        layer = layer_adapter.layer
        cross_attn = layer.layer[1].EncDecAttention  # Access cross-attention directly
        
        # Only slice K and V (they process encoder outputs)
        # Q should NOT be sliced (it processes decoder hidden states at 768-dim)
        for param_name in ['k', 'v']:
            W = getattr(cross_attn, param_name)
            
            if W.weight.shape[1] != enc_output_dim:
                # Get current parameters
                old_weight = W.weight.data
                old_bias = W.bias.data if W.bias is not None else None
                old_out_features = W.out_features
                device = old_weight.device
                dtype = old_weight.dtype
                
                # Create new Linear layer with correct input dimension
                new_linear = nn.Linear(
                    in_features=enc_output_dim,
                    out_features=old_out_features,
                    bias=(old_bias is not None),
                    device=device,
                    dtype=dtype
                )
                
                # Copy sliced weights
                new_linear.weight.data = old_weight[:, :enc_output_dim].contiguous()
                if old_bias is not None:
                    new_linear.bias.data = old_bias.contiguous()
                
                # Replace the module
                setattr(cross_attn, param_name, new_linear)
                logging.info(f"  Layer {j}: Sliced cross-attn {param_name.upper()} to {enc_output_dim}")
    
    # -------------------------
    # 6) Save configuration
    # -------------------------
    
    model_adapter.slicing_conf = SlicingConfig()
    model_adapter.slicing_conf.const_dimension = target_dim
    
    # Record encoder dimensions
    for idx in range(len(enc_layers)):
        model_adapter.slicing_conf.attention_input_dimensions[str(idx)] = target_dim
        model_adapter.slicing_conf.attention_output_dimensions[str(idx)] = target_dim
        model_adapter.slicing_conf.mlp_input_dimensions[str(idx)] = target_dim
        model_adapter.slicing_conf.mlp_output_dimensions[str(idx)] = target_dim
    
    # Decoder stays original dimension (not recorded in config)
    
    # Sanitize shortcuts
    for la in enc_layers:
        lyr = la.layer
        if hasattr(lyr, "attn_shortcut_Q"):
            lyr.attn_shortcut_Q = nn.Parameter(_sanitize_Q(lyr.attn_shortcut_Q.data, "enc_attn").cpu())
        if hasattr(lyr, "mlp_shortcut_Q"):
            lyr.mlp_shortcut_Q = nn.Parameter(_sanitize_Q(lyr.mlp_shortcut_Q.data, "enc_mlp").cpu())
    
    logging.info("="*70)
    logging.info("✓ Encoder-only slicing complete!")
    logging.info(f"  Encoder: {orig_dim} → {target_dim} dims ({(1-target_dim/orig_dim)*100:.1f}% reduction)")
    logging.info(f"  Decoder: {orig_dim} dims (unchanged)")
    logging.info(f"  Embeddings: {orig_dim} dims (unchanged)")
    logging.info("="*70)