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
        # If it has integer keys 0..N-1, use them (original behavior)
        if all(i in new_embedding_dimensions for i in range(len(embeddings))):
            dims = [new_embedding_dimensions[i] for i in range(len(embeddings))]
        else:
            # Fallback ONLY to avoid KeyError:
            # infer target dim from slicing_conf (works for T5 reloads)
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
        rotate_and_slice_parallel(
            model_adapter,
            dataloader,
            slicing_scheduler,
            apply_mask=apply_mask,
            final_orientation=final_orientation,
        )
    else:
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

        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q)
        slice_attention_output(
            layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        )

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(inps, ignore_masks)
        if final_orientation == 'random':
            
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0]).to(device=Q.device, dtype=Q.dtype)
            Q = Q @ R

        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        rotate_mlp_output(layer_adapter, Q)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

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
        return int(m.get(str(i), default))

    
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
        raise ValueError("slicing_conf.const_dimension is None")

    # 1) Slice embeddings (T5 shared)
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

    # 4) Slice pre-head layernorm (decoder final LN) to match **lm_head in_features**
    # In your setup lm_head is tied to shared, so it must be emb_dim.
    pre_head_ln = model_adapter.get_pre_head_layernorm()
    if pre_head_ln is not None and hasattr(pre_head_ln, "weight") and pre_head_ln.weight is not None:
        pre_head_ln.weight.data = pre_head_ln.weight.data[:emb_dim].contiguous()
        if getattr(pre_head_ln, "bias", None) is not None and pre_head_ln.bias is not None:
            pre_head_ln.bias.data = pre_head_ln.bias.data[:emb_dim].contiguous()

    # 5) T5 critical: slice + tie lm_head to shared
    if hasattr(model_adapter, "get_lm_head"):
        lm_head = model_adapter.get_lm_head()
        W = lm_head.weight.data[:, :emb_dim].contiguous()
        lm_head.weight = torch.nn.Parameter(W)
        lm_head.in_features = emb_dim

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

    # -------------------------
    # 2) Encoder stack (FIXED: enforce PCA space == cur_dim)
    # -------------------------
    cur_dim = int(model_adapter.model.shared.weight.shape[1])  # 688 after embedding slice
    Q_enc = torch.eye(cur_dim, device=config.device, dtype=torch.float64)

    for idx, layer_adapter in enumerate(tqdm(enc_layers, desc="Rotating/slicing encoder", unit="layer")):
        layer = layer_adapter.layer

        # ---- scheduler dims (clamped)
        d_sa_in  = min(int(enc_sched.get_attention_input_dimension(idx)), cur_dim)
        d_sa_out = min(int(enc_sched.get_attention_output_dimension(idx, match_head_dim=False)), cur_dim)
        d_mlp_in = min(int(enc_sched.get_mlp_input_dimension(idx)), d_sa_out)
        d_mlp_out = min(int(enc_sched.get_mlp_output_dimension(idx)), int(model_adapter.model.shared.weight.shape[1]))

        # ============================================================
        # 2.1 Self-attn INPUT: SLICE FIRST, then ROTATE (prevents 768x768 @ 688x688)
        # ============================================================
        slice_attention_inputs(layer_adapter, d_sa_in)

        # rotate only in the actually-kept input space
        Q_in = Q_enc[:d_sa_in, :d_sa_in].contiguous()
        rotate_attention_inputs(layer_adapter, Q_in)

        # Update hidden_states for this layer: rotate in cur_dim, then slice to d_sa_in
        for i, inp in enumerate(enc_inps):
            hs = (inp.to(config.device, dtype=dtype) @ Q_enc.to(config.device, dtype=dtype))[:, :, :d_sa_in].cpu()
            enc_args[i] = layer_adapter.get_updated_args(hs, enc_args[i])

        # ============================================================
        # 2.2 Self-attn OUTPUT: SLICE OUTPUT to d_sa_out BEFORE PCA signals
        # This ensures the layer forward produces activations in <= cur_dim space.
        # ============================================================
        slice_attention_output(layer_adapter, d_sa_out)

        # Set shortcut to map residual (cur_dim) -> d_sa_out safely
        # (rectangular identity, finite)
        layer.attn_shortcut_Q = nn.Parameter(
            torch.eye(min(cur_dim, d_sa_out), dtype=dtype).to("cpu")
            if cur_dim == d_sa_out else
            torch.nn.functional.pad(torch.eye(min(cur_dim, d_sa_out), dtype=dtype), (0, max(0, d_sa_out - cur_dim), 0, max(0, cur_dim - d_sa_out))).to("cpu")[:cur_dim, :d_sa_out]
        )
        # FFN input must match current hidden dim (cur_dim)
        slice_mlp_input(layer_adapter, cur_dim)

        _force_layer_on_device_inplace(layer, dtype)

        # Now these signals are in d_sa_out space (NOT 768)
        mlp_ln_inputs, enc_outs_tmp = get_signals(layer_adapter, enc_args, enc_kwargs)

        # PCA in the *current output* space (d_sa_out)
        _, Q_mid = pca_calc(mlp_ln_inputs, enc_ignore_masks)
        Q_mid = Q_mid.to(config.device, dtype=torch.float64)               # (d_sa_out, d_sa_out)

        # Rotate attn output in-place (square, safe)
        rotate_attention_output(layer_adapter, Q_mid)

        # ============================================================
        # 2.3 MLP INPUT: SLICE FIRST then ROTATE in d_mlp_in subspace
        # ============================================================
        slice_mlp_input(layer_adapter, d_mlp_in)
        Q_mlp_in = Q_mid[:d_mlp_in, :d_mlp_in].contiguous()
        rotate_mlp_input(layer_adapter, Q_mlp_in)

        # ============================================================
        # 2.4 MLP OUTPUT: slice to d_mlp_out and update cached activations
        # ============================================================
        slice_mlp_output(layer_adapter, d_mlp_out)

        # Run once again to get final hidden states for next layer
        _force_layer_on_device_inplace(layer, dtype)
        _, enc_inps_next = get_signals(layer_adapter, enc_args, enc_kwargs)

        # enc_inps_next are in d_mlp_out space now
        enc_inps = [x[:, :, :d_mlp_out].cpu() for x in enc_inps_next]
        for i in range(len(enc_args)):
            enc_args[i] = layer_adapter.get_updated_args(enc_inps[i], enc_args[i])

        # Next layer works in d_mlp_out
        cur_dim = d_mlp_out
        Q_enc = torch.eye(cur_dim, device=config.device, dtype=torch.float64)

        layer.to("cpu")
        cleanup_memory()

    # Encoder outputs
    enc_outs = enc_inps
    enc_kv_dim = enc_outs[0].shape[-1]


    # -------------------------
    # 3) Decoder stack (dimension-correct, no crashes)
    # -------------------------
    def rect_eye(r: int, c: int, *, dtype, device="cpu"):
        m = torch.zeros((r, c), dtype=dtype, device=device)
        k = min(r, c)
        m[:k, :k] = torch.eye(k, dtype=dtype, device=device)
        return m

    dec_inps, dec_args, dec_kwargs, dec_ignore_masks = [], [], [], []
    for batch, enc_h in zip(dec_batches, enc_outs):
        inp0, args0, kwargs0 = get_decoder_layer0_inputs(model_adapter, batch, encoder_hidden_states=enc_h)
        dec_inps.append(inp0)
        dec_args.append(args0)
        dec_kwargs.append(kwargs0)
        if apply_mask:
            dec_ignore_masks.append(batch.get("decoder_attention_mask", batch["attention_mask"]))

    enc_kv_dim = int(enc_outs[0].shape[-1])  # final encoder stream dim (should be 688)

    # In rotate.py, replace the decoder loop in rotate_and_slice_seq2seq
    # Starting around line 1093

    for j, layer_adapter in enumerate(tqdm(dec_layers, desc="Rotating/slicing decoder", unit="layer")):
        layer = layer_adapter.layer
        cur_d = int(dec_inps[0].shape[-1])   # current decoder stream dim entering this block

        # pick target dims, clamp to reality
        d_sa_in   = min(int(dec_sched.get_attention_input_dimension(j)), cur_d)
        d_sa_out  = min(int(dec_sched.get_attention_output_dimension(j, match_head_dim=False)), d_sa_in)
        d_mlp_in  = min(int(dec_sched.get_mlp_input_dimension(j)), d_sa_out)
        d_mlp_out = min(int(dec_sched.get_mlp_output_dimension(j)), d_mlp_in, int(emb_dim))

        # -------------------------
        # Self-attn: CRITICAL FIX
        # The issue: we need to handle Q/K/V head dimension vs embedding dimension carefully
        # -------------------------
        
        # Get attention components
        sa = layer_adapter.layer.layer[0].SelfAttention
        
        # IMPORTANT: T5 attention uses d_model for queries/keys/values internally
        # The input projection maps from hidden_dim -> d_model
        # We need to slice BOTH the input dimension AND ensure internal dims match
        
        # 1) Slice attention INPUTS (Q/K/V weight matrices)
        slice_attention_inputs(layer_adapter, d_sa_in)
        rotate_attention_inputs(layer_adapter, torch.eye(d_sa_in, device=config.device, dtype=torch.float64))
        
        # 2) CRITICAL: After slicing Q/K/V inputs, the attention mechanism produces
        #    outputs in the CURRENT d_model space (which is d_sa_in after slicing)
        #    So the output projection must accept d_sa_in inputs
        
        # Get the actual d_model used by attention after slicing Q/K/V
        actual_attn_d_model = sa.q.weight.shape[0]  # output features of Q projection
        
        # 3) Slice attention OUTPUT to d_sa_out rows
        slice_attention_output(layer_adapter, d_sa_out)
        
        # 4) CRITICAL FIX: Set output projection input dimension to match attention's d_model
        W_o = layer_adapter.get_attention_output()
        W_o.weight.data = W_o.weight.data[:, :actual_attn_d_model].contiguous()
        W_o.in_features = actual_attn_d_model
        
        # 5) Rotate output (this is safe now that dimensions match)
        rotate_attention_output(layer_adapter, torch.eye(d_sa_out, device=config.device, dtype=torch.float64))

        # residual shortcut: from cur_d -> d_sa_out
        layer.attn_shortcut_Q = nn.Parameter(rect_eye(cur_d, d_sa_out, dtype=dtype, device="cpu"))

        # -------------------------
        # Cross-attn inputs
        # -------------------------
        slice_cross_attention_inputs(layer_adapter, new_q_dim=d_sa_out, new_kv_dim=enc_kv_dim)
        rotate_cross_attention_inputs(layer_adapter, torch.eye(d_sa_out, device=config.device, dtype=torch.float64))

        # Cross-attn output
        slice_cross_attention_output(layer_adapter, d_mlp_in)
        rotate_cross_attention_output(layer_adapter, torch.eye(d_mlp_in, device=config.device, dtype=torch.float64))

        # CRITICAL FIX: Cross-attention output columns must match its internal d_model
        ca = layer_adapter.layer.layer[1].EncDecAttention
        actual_cross_d_model = ca.q.weight.shape[0]
        W_co = layer_adapter.get_cross_attention_output()
        W_co.weight.data = W_co.weight.data[:, :actual_cross_d_model].contiguous()
        W_co.in_features = actual_cross_d_model

        layer.cross_attn_shortcut_Q = nn.Parameter(rect_eye(d_sa_out, d_mlp_in, dtype=dtype, device="cpu"))

        # -------------------------
        # MLP
        # -------------------------
        slice_mlp_input(layer_adapter, d_mlp_in)
        rotate_mlp_input(layer_adapter, torch.eye(d_mlp_in, device=config.device, dtype=torch.float64))

        slice_mlp_output(layer_adapter, d_mlp_out)
        rotate_mlp_output(layer_adapter, torch.eye(d_mlp_out, device=config.device, dtype=torch.float64))

        layer.mlp_shortcut_Q = nn.Parameter(rect_eye(d_mlp_in, d_mlp_out, dtype=dtype, device="cpu"))

        # run once to update dec_inps
        _force_layer_on_device_inplace(layer, dtype)
        _, dec_inps = get_signals(layer_adapter, dec_args, dec_kwargs)

        next_dim = int(dec_inps[0].shape[-1])
        for i, inp in enumerate(dec_inps):
            dec_args[i] = layer_adapter.get_updated_args(inp[:, :, :next_dim].cpu(), dec_args[i])

        layer.to("cpu")
        cleanup_memory()

        # record slicing config
        dec_sched.slicing_conf.attention_input_dimensions[str(j)]  = int(d_sa_in)
        dec_sched.slicing_conf.attention_output_dimensions[str(j)] = int(d_sa_out)
        dec_sched.slicing_conf.mlp_input_dimensions[str(j)]        = int(d_mlp_in)
        dec_sched.slicing_conf.mlp_output_dimensions[str(j)]       = int(d_mlp_out)



    # -------------------------
    # 4) finalize slicing_conf + sanitize Q buffers
    # -------------------------
    model_adapter.slicing_conf = dec_sched.slicing_conf.clone()
    model_adapter.slicing_conf.const_dimension = int(emb_dim)

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
    
    # -------------------------
    # CRITICAL: Force tie lm_head to shared after all slicing operations
    # -------------------------
    model = model_adapter.model
    if hasattr(model, "shared") and hasattr(model, "lm_head"):
        d = int(model.shared.weight.shape[1])
        # Ensure lm_head has correct input dimension
        if model.lm_head.weight.shape[1] != d:
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data[:, :d].contiguous())
            model.lm_head.in_features = d
        # Tie the weights (this is the critical step)
        model.lm_head.weight = model.shared.weight
        logging.info(f"✓ Tied lm_head to shared embedding (dim={d})")



