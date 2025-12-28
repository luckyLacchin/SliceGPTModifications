# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .layernorm_fusion import fuse_modules, replace_layers
from .model_adapter import ModelAdapter, SlicingConfig
from .rotate import slice_rotated_model


def do_not_initialize(func):
    """
    A decorator that prevents initialization of torch.nn modules.
    """

    def skip(*args, **kwargs) -> None:
        pass

    def wrapper(*args, **kwargs):
        kaiming_fn = torch.nn.init.kaiming_uniform_
        uniform_fn = torch.nn.init.uniform_
        normal_fn = torch.nn.init.normal_

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        result = func(*args, **kwargs)

        torch.nn.init.kaiming_uniform_ = kaiming_fn
        torch.nn.init.uniform_ = uniform_fn
        torch.nn.init.normal_ = normal_fn

        return result

    return wrapper


@do_not_initialize
def get_model_and_tokenizer(
    model_name: str,
    model_path: str | None = None,
    *,
    uninitialized: bool = False,
    dtype: torch.dtype = torch.float16,
    token: str | bool | None = None,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the model and the tokenizer from the given path.
    Set uninitialized to True when loading a pre-rotated and sliced model; in this case no weights are loaded
    in this method.
    The corresponding model adapter class must be imported before calling this method.
    Scenarios:
    - Rotate & slice HF model: model_name = name, model_path = empty, uninitialized = False
        -> Obtain the model config and weights from HF through path = name.
        -> Ignore model_path if provided.
    - Slice pre-rotated HF model: model_name = name, model_path = empty or local path, uninitialized = True
        -> Obtain the model config from HF via path = name and create uninitialized model.
        -> If the model_path is provided, confirm this use case by checking that config.json does not exist.
        -> There are no other uses of model_path in this case.
    - Rotate & slice local model: model_name = name, model_path = local path, uninitialized = False
        -> Obtain the model config through path, and the pretrained weights from the local path.
        -> Use the model name only to determine the correct model adapter to use.
    - Slice pre-rotated local model: model_name = name, model_path = local path, uninitialized = True
        -> Obtain the model config from the local path and create an uninitialized model.
        -> Use the model name only to determine the correct model adapter to use.
        -> Confirm this case by checking that config.json exists.
    """
    model_type = "uninitialized" if uninitialized else "pretrained"
    local_model = model_path is not None

    if local_model and uninitialized:
        local_model = (pathlib.Path(model_path) / "config.json").exists()

    # for HF models the path to use is the model name
    if not local_model:
        model_path = model_name

    logging.info(
        "Loading %s config %s from %s",
        model_name,
        "and model weights" if not uninitialized else "",
        model_path if local_model else "Hugging Face",
    )

    # 1) Load model (adapter decides correct class)
    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=model_path,
        model_type=model_type,
        dtype=dtype,
        local_files_only=local_model,
        token=token,
    )

    # 2) Load tokenizer BEFORE using it
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        token=token,
        local_files_only=local_model,
    )

    # Optional: avoid huge sentinel model_max_length values breaking seq len logic
    if getattr(tokenizer, "model_max_length", None) is not None and tokenizer.model_max_length > 10**9:
        tokenizer.model_max_length = 512

    # 3) Set seqlen safely (T5 doesn't have max_position_embeddings)
    model = model_adapter.model
    model.seqlen = _infer_seqlen(model, tokenizer)
    model.eval()  # switches off dropout
    model_adapter.use_cache = False

    # 4) Adapter-specific post init (may register hooks, set pads, etc.)
    model_adapter.post_init(tokenizer)

    logging.info("Loading model done")
    return model_adapter, tokenizer


@do_not_initialize
def load_sliced_model(
    model_name: str,
    sliced_model_path: str,
    *,
    token: str | None = None,
    lora_config: Any = None,
    sparsity: float | None = None,
    round_interval: int | None = 1,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load a sliced model + tokenizer from `sliced_model_path`.
    Fixed to handle T5 cross-attention K/V weights that may be stored at full dimension.
    """
    import pathlib
    import torch
    import logging

    if sparsity is None:
        raise ValueError("sparsity must be provided to locate checkpoint/config names.")

    my_model_suffix = pathlib.Path(model_name).name
    my_ckpt_name = f"{my_model_suffix}_{sparsity}.pt"
    my_cfg_name  = f"{my_model_suffix}_{sparsity}.json"

    # 1) uninitialized skeleton
    model_adapter, tokenizer = get_model_and_tokenizer(
        model_name,
        model_path=sliced_model_path,
        uninitialized=True,
        token=token,
    )
    replace_layers(model_adapter)
    fuse_modules(model_adapter)

    orig_dim = int(model_adapter.hidden_size)

    # 2) load/derive slicing_conf
    cfg_path = pathlib.Path(sliced_model_path) / my_cfg_name
    if cfg_path.exists():
        model_adapter.slicing_conf = SlicingConfig.from_json_string(cfg_path.read_text())
    else:
        model_adapter.slicing_conf = None

    if model_adapter.slicing_conf is None:
        target_dim = int((1.0 - float(sparsity)) * orig_dim)
        target_dim -= target_dim % int(round_interval or 1)
        sc = SlicingConfig()
        sc.const_dimension = target_dim
        model_adapter.slicing_conf = sc
    else:
        if model_adapter.slicing_conf.const_dimension is None:
            raise ValueError("slicing_conf exists but const_dimension is None.")
        target_dim = int(model_adapter.slicing_conf.const_dimension)

    # 3) slice skeleton
    slice_rotated_model(model_adapter)
    model = model_adapter.model

    # 4) enforce T5 invariant: lm_head tied to shared
    if hasattr(model, "shared") and hasattr(model, "lm_head"):
        d = int(model.shared.weight.shape[1])
        if model.lm_head.weight.shape[1] != d:
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data[:, :d].contiguous())
            model.lm_head.in_features = d
        model.lm_head.weight = model.shared.weight

    # 5) optional LoRA
    if lora_config:
        from peft import get_peft_model
        model_adapter.model = get_peft_model(model_adapter.model, lora_config)
        model = model_adapter.model

    # 6) load checkpoint
    ckpt_path = pathlib.Path(sliced_model_path) / my_ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info(f"Loading sliced model weights from {sliced_model_path}")
    state = torch.load(str(ckpt_path), map_location="cpu")

    # drop shortcut_Q keys
    shortcut_suffixes = ("attn_shortcut_Q", "mlp_shortcut_Q", "cross_attn_shortcut_Q")
    for k in [k for k in list(state.keys()) if k.endswith(shortcut_suffixes)]:
        state.pop(k, None)

    # 7) FIX: Handle cross-attention K/V weights specially for T5
    # These may be stored at full encoder dimension (768) but model expects sliced encoder output dim
    model_state = model.state_dict()
    cross_attn_kv_keys = []
    for k in state.keys():
        # Match decoder cross-attention K/V weights
        if "decoder.block" in k and "layer.1.EncDecAttention" in k and (".k.weight" in k or ".v.weight" in k):
            cross_attn_kv_keys.append(k)
    
    # Determine encoder final dimension from actual model state
    enc_final_dim = target_dim  # default fallback
    
    # Method 1: Check what dimension the decoder cross-attn K/V actually expect
    if cross_attn_kv_keys and cross_attn_kv_keys[0] in model_state:
        enc_final_dim = int(model_state[cross_attn_kv_keys[0]].shape[1])
        logging.info(f"Detected encoder output dimension: {enc_final_dim} from model skeleton")
    
    # Method 2: Fallback to slicing_conf if available
    elif model_adapter.slicing_conf and hasattr(model_adapter.slicing_conf, "mlp_output_dimensions"):
        mlp_out = model_adapter.slicing_conf.mlp_output_dimensions
        if mlp_out and isinstance(mlp_out, dict) and len(mlp_out) > 0:
            try:
                # Find the highest numbered layer (last encoder layer)
                int_keys = sorted([int(k) for k in mlp_out.keys()])
                if int_keys:
                    max_idx = int_keys[-1]
                    enc_final_dim = int(mlp_out.get(str(max_idx), mlp_out.get(max_idx, target_dim)))
                    logging.info(f"Using encoder output dimension from config: {enc_final_dim}")
            except (ValueError, KeyError, TypeError) as e:
                logging.warning(f"Could not parse mlp_output_dimensions: {e}, using {target_dim}")
    
    logging.info(f"Cross-attention K/V will be sliced to encoder output dim: {enc_final_dim}")
    
    # Slice cross-attention K/V weights to match encoder output
    for k in cross_attn_kv_keys:
        if k in model_state:
            ckpt_shape = state[k].shape
            model_shape = model_state[k].shape
            
            if ckpt_shape[1] != model_shape[1]:  # in_features mismatch
                logging.info(f"Slicing {k} from {ckpt_shape} to {model_shape} (cross-attn KV)")
                # Slice in_features dimension to match encoder output
                state[k] = state[k][:, :model_shape[1]].contiguous()

    # 8) detect REAL shape mismatches (after fixing cross-attn)
    mism = []
    for k, v in state.items():
        if k in model_state and tuple(v.shape) != tuple(model_state[k].shape):
            mism.append((k, tuple(v.shape), tuple(model_state[k].shape)))

    if mism:
        preview = "\n".join([f"  - {k}: ckpt {a} vs model {b}" for k, a, b in mism[:25]])
        raise RuntimeError(
            "Checkpoint incompatible with current slicing skeleton (shape mismatches).\n"
            "You MUST regenerate the sliced model with the same rotate/slice code + slicing_conf.\n"
            f"First mismatches:\n{preview}"
        )

    # 9) load non-strict (we removed shortcuts)
    missing, unexpected = model.load_state_dict(state, strict=False)

    missing_non_shortcut = [k for k in missing if not k.endswith(shortcut_suffixes)]
    unexpected_non_shortcut = [k for k in unexpected if not k.endswith(shortcut_suffixes)]
    if missing_non_shortcut or unexpected_non_shortcut:
        raise RuntimeError(
            "Unexpected missing/unexpected keys after loading.\n"
            f"Missing (non-shortcut): {missing_non_shortcut[:50]}\n"
            f"Unexpected (non-shortcut): {unexpected_non_shortcut[:50]}"
        )

    # 10) recreate shortcuts as identity
    def _eye(n: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.eye(n, dtype=torch.float32, device="cpu"))

    enc_layers = model_adapter.get_encoder_layers() if hasattr(model_adapter, "get_encoder_layers") else model_adapter.get_layers()
    for la in enc_layers:
        layer = la.layer
        layer.attn_shortcut_Q = _eye(target_dim)
        if hasattr(layer, "mlp_shortcut_Q"):
            layer.mlp_shortcut_Q = _eye(target_dim)

    if hasattr(model_adapter, "get_decoder_layers"):
        for la in model_adapter.get_decoder_layers():
            layer = la.layer
            layer.attn_shortcut_Q = _eye(target_dim)
            if hasattr(layer, "cross_attn_shortcut_Q"):
                layer.cross_attn_shortcut_Q = _eye(target_dim)
            if hasattr(layer, "mlp_shortcut_Q"):
                layer.mlp_shortcut_Q = _eye(target_dim)

    model.eval()

    # 11) final finite check
    for n, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            raise RuntimeError(f"Non-finite parameter after load: {n}")

    return model_adapter, tokenizer


def _infer_seqlen(model, tokenizer=None, default=512) -> int:
    cfg = getattr(model, "config", None)

    # Common for decoder-only (OPT/GPT)
    if cfg is not None and hasattr(cfg, "max_position_embeddings"):
        v = getattr(cfg, "max_position_embeddings")
        if v is not None:
            return int(v)

    # Common for T5-like / seq2seq configs
    if cfg is not None and hasattr(cfg, "n_positions"):
        v = getattr(cfg, "n_positions")
        if v is not None:
            return int(v)

    # Last resort: tokenizer limit (works well for T5)
    if tokenizer is not None:
        v = getattr(tokenizer, "model_max_length", None)
        # HF sometimes sets a huge sentinel like 1e30 for "infinite"
        if v is not None and v < 10**9:
            return int(v)

    return int(default)