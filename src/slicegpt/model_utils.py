from __future__ import annotations
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Tuple, Dict

import torch
from torch import Tensor

from slicegpt.modules import RMSN

from . import utils
from .config import config
from .model_adapter import LayerAdapter, ModelAdapter


from transformers.modeling_outputs import BaseModelOutput



def get_layer0_inputs(model_adapter: ModelAdapter, batch: Tensor) -> tuple[Tensor, tuple, dict[str, Any]]:
    """
    Returns the inputs to the first layer of the model (after embeddings).

    Also returns the additional args and kwargs that are passed to
    the first layer (such as the attention mask, or caches K/V values).

    This relies on all arguments to subsequent layers being the same.

    NB: this won't work from OPT 350m.
    """
    # Move embeddings to device.
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to(config.device))

    class Catcher(torch.nn.Module):
        def __init__(self, original_layer):
            super().__init__()
            self.original_layer = original_layer

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

        def __getattr__(self, name):
            # Pass through attributes from original layer (e.g., attention_type for Gemma 3)
            if name in ('original_layer', 'saved_args', 'saved_kwargs'):
                return super().__getattr__(name)
            try:
                return super().__getattr__(name)
            except AttributeError:
                if hasattr(self, 'original_layer'):
                    return getattr(self.original_layer, name)
                raise

    layer0_adapter = model_adapter.get_layers()[0]
    layer0_catcher = Catcher(layer0_adapter.layer)
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        batch = utils.map_tensors(batch, device=config.device)
        model_adapter.model(**batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    args = utils.map_tensors(args, device='cpu')
    kwargs = utils.map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # Move embeddings back to cpu, and clear GPU cache.
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to('cpu'))

    # Run GC and cleanup GPU memory
    utils.cleanup_memory()

    return args[layer0_adapter.hidden_states_args_position], args, kwargs


def get_signals(
    layer_adapter: LayerAdapter, 
    layer_args: list[tuple], 
    layer_kwargs: list[dict[str, Any]]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Take the input signals ("activations") for a layer, run the layer forward.
    Return the output of the layer (not layernormed) and the input to the MLP (pre-layernorm).
    
    For T5 decoder layers, this properly handles encoder_hidden_states for cross-attention.
    """
    mlp_ln_inputs = []
    outputs = []

    layer_adapter.layer.to(config.device)

    def hook_fn(_, args: tuple, _output: Any) -> None:
        inp = args[0]  # Position in RMSN.forward args
        mlp_ln_inputs.append(inp.cpu())

    second_layernorm = layer_adapter.get_second_layernorm()
    assert isinstance(second_layernorm, RMSN)
    hook = second_layernorm.register_forward_hook(hook_fn)
    
    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        
        # CRITICAL FIX: For T5 decoder, ensure encoder_hidden_states is present
        # If it's missing, the cross-attention will fail and produce garbage
        if layer_adapter.has_cross_attention:
            # Check if encoder_hidden_states is already in kwargs
            if 'encoder_hidden_states' not in layer_kwargs_batch:
                raise RuntimeError(
                    "T5 decoder layer requires encoder_hidden_states in kwargs for cross-attention. "
                    "Make sure to pass encoder outputs when calling get_signals() on decoder layers."
                )
        
        out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
        if isinstance(out, tuple):
            out = out[layer_adapter.hidden_states_output_position]
        out = out.cpu()
        outputs.append(out)

        if mlp_ln_inputs[i].ndim == 2:
            batch_size, seqlen, _ = out.shape
            mlp_ln_inputs[i] = mlp_ln_inputs[i].reshape(batch_size, seqlen, -1)

    hook.remove()
    return mlp_ln_inputs, outputs

def get_cross_attn_ln_inputs(
    layer_adapter: LayerAdapter,
    layer_args: list[tuple],
    layer_kwargs: list[dict[str, Any]],
) -> list[torch.Tensor]:
    """
    Run decoder layer forward and capture the inputs to the cross-attention layer_norm
    (i.e., the LN right before EncDecAttention).

    Returns list of tensors (one per batch item), on CPU.
    """
    cross_ln_inputs: list[torch.Tensor] = []
    layer_adapter.layer.to(config.device)

    # This requires your decoder adapter to expose get_cross_attention_layernorm()
    cross_ln = layer_adapter.get_cross_attention_layernorm()
    assert isinstance(cross_ln, RMSN), "Expected RMSN after LN->RMSN fusion"

    def hook_fn(_module, args: tuple, _output: Any) -> None:
        inp = args[0]  # RMSN.forward first arg
        cross_ln_inputs.append(inp.cpu())

    hook = cross_ln.register_forward_hook(hook_fn)

    for i, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(layer_args, layer_kwargs)):
        layer_args_batch, layer_kwargs_batch = utils.map_tensors(
            [layer_args_batch, layer_kwargs_batch], device=config.device
        )
        _ = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)

        # reshape to (B, T, D) if it came as (B*T, D) (rare, but keep symmetry with get_signals)
        if cross_ln_inputs[i].ndim == 2:
            out_hidden = layer_args_batch[layer_adapter.hidden_states_args_position]
            batch_size, seqlen, _ = out_hidden.shape
            cross_ln_inputs[i] = cross_ln_inputs[i].reshape(batch_size, seqlen, -1)

    hook.remove()
    return cross_ln_inputs

''''
def get_ln_inputs(layer_adapter, which: str, layer_args, layer_kwargs):
    ln = layer_adapter.get_cross_attention_layernorm() if which=="cross" else layer_adapter.get_second_layernorm()
    hook input collection exactly like get_signals
'''


def _move_embeddings_to_device(model_adapter: ModelAdapter, device: str) -> None:
    # Same pattern as get_layer0_inputs(): move only embeddings to GPU so the forward reaches layer0.
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to(device))


def _move_embeddings_to_cpu(model_adapter: ModelAdapter) -> None:
    for W in model_adapter.get_embeddings():
        W.weight = torch.nn.Parameter(W.weight.to("cpu"))


def get_encoder_layer0_inputs(
    model_adapter: ModelAdapter,
    batch: Dict[str, Tensor],
) -> Tuple[Tensor, tuple, Dict[str, Any]]:
    """
    Returns the inputs to encoder block 0 (after shared embeddings),
    plus the full args/kwargs captured for encoder block 0.

    Requires ModelAdapter to implement:
      - get_encoder_layers()
      - get_raw_encoder_layer_at(i)
      - set_raw_encoder_layer_at(i, new_layer)
    """
    _move_embeddings_to_device(model_adapter, config.device)

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.saved_args = None
            self.saved_kwargs = None

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError("encoder catcher")

    # Take layer0 adapter just to know hidden_states position and to restore the real layer
    layer0_adapter = model_adapter.get_encoder_layers()[0]
    catcher = Catcher()

    # Swap encoder block 0
    original_layer0 = model_adapter.get_raw_encoder_layer_at(0)
    model_adapter.set_raw_encoder_layer_at(0, catcher)

    try:
        batch_dev = utils.map_tensors(batch, device=config.device)
        # Run full model; it will enter encoder block 0 and we intercept there.
        model_adapter.model(**batch_dev)
    except ValueError:
        pass
    finally:
        # Restore encoder block 0
        model_adapter.set_raw_encoder_layer_at(0, original_layer0)

    if catcher.saved_args is None:
        raise RuntimeError(
            "Encoder catcher did not trigger. "
            "Check that model_adapter.set_raw_encoder_layer_at(0, ...) targets encoder.block[0]."
        )

    args = utils.map_tensors(catcher.saved_args, device="cpu")
    kwargs = utils.map_tensors(catcher.saved_kwargs, device="cpu")

    _move_embeddings_to_cpu(model_adapter)
    utils.cleanup_memory()

    hidden_states = args[layer0_adapter.hidden_states_args_position]
    return hidden_states, args, kwargs

def _ensure_decoder_input_ids(model, batch_dev: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Ensure batch has decoder_input_ids. Prefer HF helper if available.
    """
    if "decoder_input_ids" in batch_dev:
        return batch_dev

    if "labels" not in batch_dev:
        raise RuntimeError("Need labels or decoder_input_ids to run decoder")

    labels = batch_dev["labels"]

    # If HF provides helper, use it
    if hasattr(model, "prepare_decoder_input_ids_from_labels"):
        batch_dev["decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(labels=labels)
        return batch_dev

    # Fallback: standard T5 shift_right
    pad_id = model.config.pad_token_id
    start_id = model.config.decoder_start_token_id
    if start_id is None:
        start_id = pad_id

    # replace -100 with pad
    labels2 = labels.clone()
    labels2 = torch.where(labels2 == -100, torch.tensor(pad_id, device=labels2.device), labels2)

    decoder_input_ids = labels2.new_zeros(labels2.shape)
    decoder_input_ids[:, 0] = start_id
    decoder_input_ids[:, 1:] = labels2[:, :-1]
    batch_dev["decoder_input_ids"] = decoder_input_ids
    return batch_dev


def get_decoder_layer0_inputs(
    model_adapter: ModelAdapter,
    batch: Dict[str, Tensor],
    *,
    encoder_hidden_states: Tensor,
) -> Tuple[Tensor, tuple, Dict[str, Any]]:
    _move_embeddings_to_device(model_adapter, config.device)

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.saved_args = None
            self.saved_kwargs = None

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError("__DECODER_CATCHER__")

    layer0_adapter = model_adapter.get_decoder_layers()[0]
    catcher = Catcher()

    original_layer0 = model_adapter.get_raw_decoder_layer_at(0)
    model_adapter.set_raw_decoder_layer_at(0, catcher)

    try:
        batch_dev = utils.map_tensors(batch, device=config.device)
        batch_dev = dict(batch_dev)

        # Ensure decoder_input_ids exists (robust across HF versions)
        batch_dev = _ensure_decoder_input_ids(model_adapter.model, batch_dev)

        # Force cross-attn
        enc_h = encoder_hidden_states.to(device=config.device)
        batch_dev["encoder_outputs"] = BaseModelOutput(last_hidden_state=enc_h)

        model_adapter.model(**batch_dev)

    except ValueError as e:
        if str(e) != "__DECODER_CATCHER__":
            # real ValueError -> propagate
            raise
    finally:
        model_adapter.set_raw_decoder_layer_at(0, original_layer0)

    if catcher.saved_args is None:
        raise RuntimeError(
            "Decoder catcher did not trigger. "
            "Verify set_raw_decoder_layer_at targets model.decoder.block[0] "
            "and that decoder_input_ids/labels are present."
        )

    args = utils.map_tensors(catcher.saved_args, device="cpu")
    kwargs = utils.map_tensors(catcher.saved_kwargs, device="cpu")

    _move_embeddings_to_cpu(model_adapter)
    utils.cleanup_memory()

    hidden_states = args[layer0_adapter.hidden_states_args_position]
    return hidden_states, args, kwargs

