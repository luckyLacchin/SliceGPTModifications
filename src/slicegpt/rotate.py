# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
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
    if getattr(model_adapter.config, "is_encoder_decoder", False):
        #rotate_and_slice_seq2seq(...)
        return #we have still to implement this fully
    elif model_adapter.parallel_blocks:
        rotate_and_slice_parallel(...)
    else:
        rotate_and_slice_sequential(...)



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

def rotate_cross_attention_inputs(layer_adapter: LayerAdapter, Q_q: torch.Tensor, Q_kv: torch.Tensor) -> None:
    # Q projection uses decoder hidden basis
    Wq = layer_adapter.get_cross_attention_q_input()
    dtype = Wq.weight.dtype
    Wq_ = Wq.weight.to(device=config.device, dtype=torch.float64)
    Wq.weight.data = torch.matmul(Wq_, Q_q).to(device="cpu", dtype=dtype)

    # K,V projections use encoder hidden basis
    for W in layer_adapter.get_cross_attention_kv_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q_kv).to(device="cpu", dtype=dtype)


def slice_cross_attention_inputs(layer_adapter: LayerAdapter, new_q_dim: int, new_kv_dim: int) -> None:
    # Q input: columns slice
    Wq = layer_adapter.get_cross_attention_q_input()
    Wq.weight.data = Wq.weight.data[:, :new_q_dim]
    Wq.in_features = new_q_dim

    # K,V input: columns slice
    for W in layer_adapter.get_cross_attention_kv_inputs():
        W.weight.data = W.weight.data[:, :new_kv_dim]
        W.in_features = new_kv_dim

    # also slice shortcut_Q "row dimension" if used later (same pattern as attn)
    layer_adapter.layer.cross_attn_shortcut_Q = torch.nn.Parameter(
        layer_adapter.layer.cross_attn_shortcut_Q[:new_q_dim, :]
    )


def rotate_cross_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    W = layer_adapter.get_cross_attention_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_cross_attention_output(layer_adapter: LayerAdapter, new_dim: int) -> None:
    W = layer_adapter.get_cross_attention_output()
    W.weight.data = W.weight.data[:new_dim, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_dim]
    W.out_features = new_dim

'''
def rotate_and_slice_seq2seq(model_adapter, dataloader, slicing_scheduler, ...):
    # 1) run encoder stack slicing: produce final Q_enc_out and store sliced encoder outputs for each batch
    # 2) run decoder slicing with cross-attn using those encoder outputs
'''

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
    dec_batches = []  # keep original batches for later decoder pass

    for batch in dataloader:
        dec_batches.append(batch)  # keep it
        inp0, args0, kwargs0 = get_encoder_layer0_inputs(model_adapter, batch)
        enc_inps.append(inp0)
        enc_args.append(args0)
        enc_kwargs.append(kwargs0)
        if apply_mask:
            enc_ignore_masks.append(batch["attention_mask"])

    # Setup scheduler as seq2seq (you'll extend it or just reuse two schedulers)
    # Practical approach: run scheduler.setup twice (encoder + decoder)
    enc_layers = model_adapter.get_encoder_layers()
    dec_layers = model_adapter.get_decoder_layers()

    slicing_scheduler.setup(
        hidden_size=model_adapter.hidden_size,
        layers_num=len(enc_layers) + len(dec_layers),
        parallel_blocks=False
    )

    # -------------------------
    # 1) Rotate/slice shared embeddings using PCA on encoder layer0 inputs
    # -------------------------
    _, Q = pca_calc(enc_inps, enc_ignore_masks)
    Q = Q.to(config.device)

    if final_orientation == "random":
        R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
        Q = Q @ R.to(Q.device)

    rotate_embeddings(model_adapter, Q)
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # -------------------------
    # 2) Encoder stack: same as rotate_and_slice_sequential
    # -------------------------
    Q_enc = Q  # current basis for encoder hidden states

    for idx, layer_adapter in enumerate(tqdm(enc_layers, desc="Rotating/slicing encoder", unit="layer")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q_enc.T.clone().to(dtype=dtype))

        rotate_attention_inputs(layer_adapter, Q_enc)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        # update inputs to this layer
        for i, inp in enumerate(enc_inps):
            enc_args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(config.device), Q_enc.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(idx)
                ].cpu(),
                enc_args[i],
            )

        # PCA at FFN boundary (your existing get_signals hooks second LN)
        mlp_ln_inputs, _ = get_signals(layer_adapter, enc_args, enc_kwargs)
        _, Q_mid = pca_calc(mlp_ln_inputs, enc_ignore_masks)
        Q_mid = Q_mid.to(config.device, dtype=torch.float64)

        if final_orientation == "random":
            R = random_orthogonal_upper_left(
                Q_mid.shape[0],
                slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False),
            )
            Q_mid = Q_mid @ R.to(Q_mid.device)

        # update attn shortcut + rotate/slice attn output
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q_mid.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q_mid)
        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False))

        # rotate/slice FFN input
        layer.mlp_shortcut_Q = nn.Parameter(
            Q_mid.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(idx), :]
        )
        rotate_mlp_input(layer_adapter, Q_mid)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))

        cleanup_memory()

        # run layer to get next signals
        _, enc_inps = get_signals(layer_adapter, enc_args, enc_kwargs)

        _, Q_next = pca_calc(enc_inps, enc_ignore_masks)
        Q_next = Q_next.to(config.device, dtype=torch.float64)

        if final_orientation == "random":
            R = random_orthogonal_upper_left(Q_next.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q_next = Q_next @ R.to(Q_next.device)

        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q_next.to(dtype=dtype)))

        rotate_mlp_output(layer_adapter, Q_next)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(idx)])

        layer.to("cpu")
        cleanup_memory()

        Q_enc = Q_next  # encoder basis progresses like decoder-only case

    # After encoder loop:
    Q_enc_out = Q_enc  # IMPORTANT: basis of encoder outputs
    enc_outs = enc_inps  # list of CPU tensors per batch (encoder outputs, already sliced)

    # -------------------------
    # 3) Decoder stack
    # -------------------------
    # We need decoder layer0 inputs (after embeddings) and its args/kwargs, but with encoder_hidden_states injected.
    dec_inps, dec_args, dec_kwargs, dec_ignore_masks = [], [], [], []

    for batch, enc_h in zip(dec_batches, enc_outs):
        inp0, args0, kwargs0 = get_decoder_layer0_inputs(
            model_adapter,
            batch,
            encoder_hidden_states=enc_h,  # already sliced encoder outputs
        )
        dec_inps.append(inp0)
        dec_args.append(args0)
        dec_kwargs.append(kwargs0)
        if apply_mask:
            dec_ignore_masks.append(batch["attention_mask"])

    # Decoder basis starts from the same embedding rotation basis
    Q_dec = Q  # decoder hidden basis

    # Note: indices in scheduler for decoder will continue after encoder or use separate scheduler
    base = len(enc_layers)

    for j, layer_adapter in enumerate(tqdm(dec_layers, desc="Rotating/slicing decoder", unit="layer")):
        layer_idx = base + j
        layer = layer_adapter.layer

        # ---- (A) Self-attention boundary ----
        layer.attn_shortcut_Q = nn.Parameter(Q_dec.T.clone().to(dtype=dtype))

        rotate_attention_inputs(layer_adapter, Q_dec)
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(layer_idx))

        # update decoder inputs for this layer
        for i, inp in enumerate(dec_inps):
            dec_args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(config.device), Q_dec.to(dtype=dtype))[
                    :, :, : slicing_scheduler.get_attention_input_dimension(layer_idx)
                ].cpu(),
                dec_args[i],
            )

        # ---- (B) PCA at cross-attn LN boundary ----
        # You need a helper like get_cross_attn_ln_inputs(layer_adapter, dec_args, dec_kwargs)
        cross_ln_inputs = get_cross_attn_ln_inputs(layer_adapter, dec_args, dec_kwargs)
        _, Q_after_self = pca_calc(cross_ln_inputs, dec_ignore_masks)
        Q_after_self = Q_after_self.to(config.device, dtype=torch.float64)

        # Update self-attn shortcut and rotate/slice self-attn output into the new basis
        layer.attn_shortcut_Q = nn.Parameter(
            torch.matmul(
                layer.attn_shortcut_Q,
                Q_after_self.to(dtype=dtype)[:, : slicing_scheduler.get_attention_output_dimension(layer_idx, match_head_dim=False)],
            )
        )
        rotate_attention_output(layer_adapter, Q_after_self)
        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(layer_idx, match_head_dim=False))

        # ---- (C) Cross-attention projections ----
        # Cross-attn Q sees decoder space (Q_after_self)
        # Cross-attn K,V see encoder space (Q_enc_out)
        layer.cross_attn_shortcut_Q = nn.Parameter(Q_after_self.T.clone().to(dtype=dtype))
        rotate_cross_attention_inputs(layer_adapter, Q_q=Q_after_self, Q_kv=Q_enc_out)
        slice_cross_attention_inputs(
            layer_adapter,
            new_q_dim=slicing_scheduler.get_attention_input_dimension(layer_idx),
            new_kv_dim=slicing_scheduler.get_mlp_input_dimension(0),  # you'll define a dedicated encoder-dim accessor
        )

        # ---- (D) PCA at FFN LN boundary ----
        # Now hook FFN LN like in get_signals()
        mlp_ln_inputs, _ = get_signals(layer_adapter, dec_args, dec_kwargs)
        _, Q_after_cross = pca_calc(mlp_ln_inputs, dec_ignore_masks)
        Q_after_cross = Q_after_cross.to(config.device, dtype=torch.float64)

        # update cross-attn shortcut and rotate/slice cross-attn output
        layer.cross_attn_shortcut_Q = nn.Parameter(torch.matmul(layer.cross_attn_shortcut_Q, Q_after_cross.to(dtype=dtype)))
        rotate_cross_attention_output(layer_adapter, Q_after_cross)
        slice_cross_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(layer_idx, match_head_dim=False))

        # ---- (E) FFN block ----
        layer.mlp_shortcut_Q = nn.Parameter(
            Q_after_cross.T.clone().to(dtype=dtype)[: slicing_scheduler.get_mlp_input_dimension(layer_idx), :]
        )
        rotate_mlp_input(layer_adapter, Q_after_cross)
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(layer_idx))

        cleanup_memory()

        # run layer and compute next PCA on outputs
        _, dec_inps = get_signals(layer_adapter, dec_args, dec_kwargs)
        _, Q_next = pca_calc(dec_inps, dec_ignore_masks)
        Q_next = Q_next.to(config.device, dtype=torch.float64)

        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q_next.to(dtype=dtype)))
        rotate_mlp_output(layer_adapter, Q_next)
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(layer_idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(layer_idx)])

        layer.to("cpu")
        cleanup_memory()

        Q_dec = Q_next

    # -------------------------
    # 4) Head
    # -------------------------
    rotate_head(model_adapter, Q_dec)
    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
