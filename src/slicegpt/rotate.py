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

    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])


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


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]


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
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
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
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)
        if final_orientation == 'random':
            R = random_orthogonal_upper_left(
                Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
            )
            Q = Q @ R.to(Q.device)

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
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

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
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)
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
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype))[
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
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)

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


def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            # backward compatibility for when no config is available
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    # slice embeddings
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # slice layers
    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        # slice attn weights 2nd dim, attn shortcut 1st dim
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))

        # slice mlp input 2nd dimension
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))

        # slice mlp shortcut 1st dimension
        # slice mlp shortcut
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])

        # slice mlp weights 1st dimension
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:  # parallel case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:  # sequential case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            )
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )

            # slice attention weights 1st dimension
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


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



def slice_cross_attention_inputs(layer_adapter, new_q_dim: int, new_kv_dim: int) -> None:
    Wq = layer_adapter.get_cross_attention_q_input()
    Wq.weight.data = Wq.weight.data[:, :new_q_dim]
    Wq.in_features = new_q_dim

    for W in layer_adapter.get_cross_attention_kv_inputs():
        W.weight.data = W.weight.data[:, :new_kv_dim]
        W.in_features = new_kv_dim

    # Slice ONLY the residual-input dimension (rows) here.
    layer = layer_adapter.layer
    if getattr(layer, "cross_attn_shortcut_Q", None) is not None:
        layer.cross_attn_shortcut_Q = nn.Parameter(layer.cross_attn_shortcut_Q[:new_q_dim, :])





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
        enc_inps.append(inp0)
        enc_args.append(args0)
        enc_kwargs.append(kwargs0)
        if apply_mask:
            enc_ignore_masks.append(batch["attention_mask"])

    enc_layers = model_adapter.get_encoder_layers()
    dec_layers = model_adapter.get_decoder_layers()

    # ======================================================================
    # Option 1 (safest): run TWO independent scheduler setups
    #   - encoder scheduler: layers_num = #encoder layers
    #   - decoder scheduler: layers_num = #decoder layers
    # The only coupling between the two stacks is the *encoder output dimension*
    # which becomes the KV input dimension for decoder cross-attention.
    # ======================================================================
    enc_sched: SlicingScheduler = copy.deepcopy(slicing_scheduler)
    dec_sched: SlicingScheduler = copy.deepcopy(slicing_scheduler)
    enc_sched.setup(hidden_size=model_adapter.hidden_size, layers_num=len(enc_layers), parallel_blocks=False)
    dec_sched.setup(hidden_size=model_adapter.hidden_size, layers_num=len(dec_layers), parallel_blocks=False)

    # -------------------------
    # 1) Rotate/slice shared embeddings using PCA on encoder layer0 inputs
    # -------------------------
    _, Q = pca_calc(enc_inps, enc_ignore_masks)
    Q = Q.to(config.device)

    emb_dims = enc_sched.get_embedding_dimensions()
    if final_orientation == "random":
        R = random_orthogonal_upper_left(Q.shape[0], emb_dims[0])
        Q = Q @ R.to(Q.device)

    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, emb_dims)
    # keep decoder scheduler embedding config consistent (shared weights)
    _ = dec_sched.get_embedding_dimensions()

    # -------------------------
    # 2) Encoder stack
    # -------------------------
    Q_enc = Q
    for idx, layer_adapter in enumerate(tqdm(enc_layers, desc="Rotating/slicing encoder", unit="layer")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q_enc.T.clone().to(dtype=dtype))

        d_sa_in = enc_sched.get_attention_input_dimension(idx)
        rotate_attention_inputs(layer_adapter, Q_enc)
        slice_attention_inputs(layer_adapter, d_sa_in)

        for i, inp in enumerate(enc_inps):
            enc_args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(config.device), Q_enc.to(dtype=dtype))[:, :, :d_sa_in].cpu(),
                enc_args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, enc_args, enc_kwargs)
        _, Q_mid = pca_calc(mlp_ln_inputs, enc_ignore_masks)
        Q_mid = Q_mid.to(config.device, dtype=torch.float64)

        d_sa_out = enc_sched.get_attention_output_dimension(idx, match_head_dim=False)
        if final_orientation == "random":
            R = random_orthogonal_upper_left(Q_mid.shape[0], d_sa_out)
            Q_mid = Q_mid @ R.to(Q_mid.device)

        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q_mid.to(dtype=dtype)[:, :d_sa_out]))
        rotate_attention_output(layer_adapter, Q_mid)
        slice_attention_output(layer_adapter, d_sa_out)

        d_mlp_in = enc_sched.get_mlp_input_dimension(idx)
        layer.mlp_shortcut_Q = nn.Parameter(Q_mid.T.clone().to(dtype=dtype)[:d_mlp_in, :])
        rotate_mlp_input(layer_adapter, Q_mid)
        slice_mlp_input(layer_adapter, d_mlp_in)

        cleanup_memory()

        _, enc_inps = get_signals(layer_adapter, enc_args, enc_kwargs)
        _, Q_next = pca_calc(enc_inps, enc_ignore_masks)
        Q_next = Q_next.to(config.device, dtype=torch.float64)

        d_mlp_out = enc_sched.get_mlp_output_dimension(idx)
        if final_orientation == "random":
            R = random_orthogonal_upper_left(Q_next.shape[0], d_mlp_out)
            Q_next = Q_next @ R.to(Q_next.device)

        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q_next.to(dtype=dtype)))
        rotate_mlp_output(layer_adapter, Q_next)
        slice_mlp_output(layer_adapter, d_mlp_out)
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, :d_mlp_out])

        layer.to("cpu")
        cleanup_memory()

        Q_enc = Q_next

    # Encoder results
    Q_enc_out = Q_enc
    enc_outs = enc_inps  # CPU tensors (B, T, Denc)
    enc_kv_dim = enc_outs[0].shape[-1]

    # -------------------------
    # 3) Decoder stack
    # -------------------------
    dec_inps, dec_args, dec_kwargs, dec_ignore_masks = [], [], [], []
    for batch, enc_h in zip(dec_batches, enc_outs):
        inp0, args0, kwargs0 = get_decoder_layer0_inputs(
            model_adapter,
            batch,
            encoder_hidden_states=enc_h,
        )
        dec_inps.append(inp0)
        dec_args.append(args0)
        dec_kwargs.append(kwargs0)
        if apply_mask:
            dec_ignore_masks.append(batch["attention_mask"])

    Q_dec = Q
    for j, layer_adapter in enumerate(tqdm(dec_layers, desc="Rotating/slicing decoder", unit="layer")):
        layer = layer_adapter.layer

        # (A) Self-attention
        layer.attn_shortcut_Q = nn.Parameter(Q_dec.T.clone().to(dtype=dtype))
        d_sa_in = dec_sched.get_attention_input_dimension(j)
        rotate_attention_inputs(layer_adapter, Q_dec)
        slice_attention_inputs(layer_adapter, d_sa_in)

        for i, inp in enumerate(dec_inps):
            dec_args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(config.device), Q_dec.to(dtype=dtype))[:, :, :d_sa_in].cpu(),
                dec_args[i],
            )

        # (B) PCA at cross-attn LN boundary
        cross_ln_inputs = get_cross_attn_ln_inputs(layer_adapter, dec_args, dec_kwargs)
        _, Q_after_self = pca_calc(cross_ln_inputs, dec_ignore_masks)
        Q_after_self = Q_after_self.to(config.device, dtype=torch.float64)

        d_sa_out = dec_sched.get_attention_output_dimension(j, match_head_dim=False)
        if final_orientation == "random":
            R = random_orthogonal_upper_left(Q_after_self.shape[0], d_sa_out)
            Q_after_self = Q_after_self @ R.to(Q_after_self.device)

        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q_after_self.to(dtype=dtype)[:, :d_sa_out]))
        rotate_attention_output(layer_adapter, Q_after_self)
        slice_attention_output(layer_adapter, d_sa_out)

        # (C) Cross-attention (KV lives in encoder space)
        d_ca_in = d_sa_out
        layer.cross_attn_shortcut_Q = nn.Parameter(Q_after_self.T.clone().to(dtype=dtype))
        rotate_cross_attention_inputs(layer_adapter, Q_dec=Q_after_self, Q_enc=Q_enc_out)
        slice_cross_attention_inputs(layer_adapter, new_q_dim=d_ca_in, new_kv_dim=enc_kv_dim)

        # (D) PCA at FFN LN boundary
        mlp_ln_inputs, _ = get_signals(layer_adapter, dec_args, dec_kwargs)
        _, Q_after_cross = pca_calc(mlp_ln_inputs, dec_ignore_masks)
        Q_after_cross = Q_after_cross.to(config.device, dtype=torch.float64)

        # We keep decoder attention-output dim consistent with the scheduler at this layer.
        d_ca_out = dec_sched.get_attention_output_dimension(j, match_head_dim=False)
        if final_orientation == "random":
            R = random_orthogonal_upper_left(Q_after_cross.shape[0], d_ca_out)
            Q_after_cross = Q_after_cross @ R.to(Q_after_cross.device)

        layer.cross_attn_shortcut_Q = nn.Parameter(
            torch.matmul(layer.cross_attn_shortcut_Q, Q_after_cross.to(dtype=dtype)[:d_ca_in, :d_ca_out])
        )
        rotate_cross_attention_output(layer_adapter, Q_after_cross)
        slice_cross_attention_output(layer_adapter, d_ca_out)

        # (E) FFN
        d_mlp_in = dec_sched.get_mlp_input_dimension(j)
        d_mlp_out = dec_sched.get_mlp_output_dimension(j)
        layer.mlp_shortcut_Q = nn.Parameter(Q_after_cross.T.clone().to(dtype=dtype)[:d_mlp_in, :])
        rotate_mlp_input(layer_adapter, Q_after_cross)
        slice_mlp_input(layer_adapter, d_mlp_in)

        cleanup_memory()

        _, dec_inps = get_signals(layer_adapter, dec_args, dec_kwargs)
        _, Q_next = pca_calc(dec_inps, dec_ignore_masks)
        Q_next = Q_next.to(config.device, dtype=torch.float64)
        if final_orientation == "random":
            R = random_orthogonal_upper_left(Q_next.shape[0], d_mlp_out)
            Q_next = Q_next @ R.to(Q_next.device)

        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q_next.to(dtype=dtype)))
        rotate_mlp_output(layer_adapter, Q_next)
        slice_mlp_output(layer_adapter, d_mlp_out)
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, :d_mlp_out])

        layer.to("cpu")
        cleanup_memory()

        Q_dec = Q_next

    # -------------------------
    # 4) Head
    # -------------------------
    rotate_head(model_adapter, Q_dec)
    if dec_sched.do_slice_head:
        slice_head(model_adapter, dec_sched.get_head_dimension())

    # -------------------------
    # 5) Save a merged slicing config (encoder indices first, decoder indices offset)
    # -------------------------
    merged = SlicingConfig(
        hidden_size=model_adapter.hidden_size,
        layers_num=len(enc_layers) + len(dec_layers),
        do_slice_head=dec_sched.do_slice_head,
        parallel_blocks=False,
    )
    merged.embedding_dimensions = enc_sched.slicing_conf.embedding_dimensions

    # encoder [0..E-1]
    merged.attention_input_dimensions.update(enc_sched.slicing_conf.attention_input_dimensions)
    merged.attention_output_dimensions.update(enc_sched.slicing_conf.attention_output_dimensions)
    merged.mlp_input_dimensions.update(enc_sched.slicing_conf.mlp_input_dimensions)
    merged.mlp_output_dimensions.update(enc_sched.slicing_conf.mlp_output_dimensions)

    # decoder offset by E
    off = len(enc_layers)
    merged.attention_input_dimensions.update({k + off: v for k, v in dec_sched.slicing_conf.attention_input_dimensions.items()})
    merged.attention_output_dimensions.update({k + off: v for k, v in dec_sched.slicing_conf.attention_output_dimensions.items()})
    merged.mlp_input_dimensions.update({k + off: v for k, v in dec_sched.slicing_conf.mlp_input_dimensions.items()})
    merged.mlp_output_dimensions.update({k + off: v for k, v in dec_sched.slicing_conf.mlp_output_dimensions.items()})

    merged.head_dimension = dec_sched.slicing_conf.head_dimension
    merged.const_dimension = getattr(slicing_scheduler.slicing_conf, "const_dimension", None)

    model_adapter.slicing_conf = merged.clone()
