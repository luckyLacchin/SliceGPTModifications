# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Any

import torch
import torch.nn as nn
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
    
    MODIFIED: Now automatically adds projection layer for encoder-only sliced T5 models.
    
    For encoder-only slicing:
    - Embeddings: original dimension (e.g., 768)
    - Encoder: sliced dimension (e.g., 688)
    - Decoder: original dimension (e.g., 768)
    
    This function detects the mismatch and adds a projection layer to bridge:
    embeddings (768) → projection → encoder (688)
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
        # Handle both const_dimension (T5) and per-layer dimensions (Gemma)
        if model_adapter.slicing_conf.const_dimension is not None:
            target_dim = int(model_adapter.slicing_conf.const_dimension)
        else:
            # For models with per-layer dimensions (e.g., Gemma), use FINAL layer's output dimension
            # This is what feeds into lm_head, so it's the critical dimension
            if hasattr(model_adapter.slicing_conf, 'mlp_output_dimensions') and model_adapter.slicing_conf.mlp_output_dimensions:
                # Get the last (highest numbered) layer's MLP output dimension
                mlp_dims = model_adapter.slicing_conf.mlp_output_dimensions
                layer_indices = sorted([int(k) for k in mlp_dims.keys()])
                if layer_indices:
                    last_layer_idx = layer_indices[-1]
                    # Try both int and str keys
                    if last_layer_idx in mlp_dims:
                        target_dim = int(mlp_dims[last_layer_idx])
                    elif str(last_layer_idx) in mlp_dims:
                        target_dim = int(mlp_dims[str(last_layer_idx)])
                    else:
                        raise KeyError(f"Cannot find layer {last_layer_idx} in mlp_output_dimensions")
                    logging.info(f"Inferred target dimension from final layer {last_layer_idx} MLP output: {target_dim}")
                else:
                    raise ValueError("mlp_output_dimensions exists but is empty")
            elif hasattr(model_adapter.slicing_conf, 'attention_input_dimensions') and model_adapter.slicing_conf.attention_input_dimensions:
                # Fallback: use first layer's dimension
                first_layer_dim = next(iter(model_adapter.slicing_conf.attention_input_dimensions.values()))
                target_dim = int(first_layer_dim)
                logging.info(f"Inferred target dimension from first layer: {target_dim}")
            else:
                # Last fallback: calculate from sparsity
                target_dim = int((1.0 - float(sparsity)) * orig_dim)
                target_dim -= target_dim % int(round_interval or 1)
                logging.warning(f"Could not find dimension in config, calculated from sparsity: {target_dim}")

    # 3) slice skeleton
    # MODIFIED: Check if this is an encoder-only sliced checkpoint by examining the checkpoint
    ckpt_path_temp = pathlib.Path(sliced_model_path) / my_ckpt_name
    is_encoder_only_slicing = False
    
    if ckpt_path_temp.exists():
        # Peek at checkpoint to detect encoder-only slicing
        state_temp = torch.load(str(ckpt_path_temp), map_location="cpu")
        
        # Check decoder dimension from checkpoint
        # If decoder weights are at original dimension (768), this is encoder-only slicing
        decoder_check_key = "decoder.block.0.layer.0.SelfAttention.q.weight"
        if decoder_check_key in state_temp:
            decoder_dim = state_temp[decoder_check_key].shape[1]  # in_features
            if decoder_dim == orig_dim:  # e.g., 768
                is_encoder_only_slicing = True
                logging.info(f"Detected encoder-only sliced checkpoint (decoder at {decoder_dim} dims)")
        
        del state_temp  # Free memory
    
    if is_encoder_only_slicing:
        # For encoder-only slicing: Only slice the encoder, leave decoder and embeddings at original dimension
        logging.info("Applying encoder-only slicing to model skeleton...")
        
        # Manually slice only encoder layers
        if hasattr(model_adapter, 'get_encoder_layers'):
            from .rotate import slice_attention_inputs, slice_attention_output, slice_mlp_input, slice_mlp_output
            
            for layer_adapter in model_adapter.get_encoder_layers():
                slice_attention_inputs(layer_adapter, target_dim)
                slice_attention_output(layer_adapter, target_dim)
                slice_mlp_input(layer_adapter, target_dim)
                slice_mlp_output(layer_adapter, target_dim)
            
            # Slice encoder final layer norm
            if hasattr(model_adapter.model.encoder, 'final_layer_norm'):
                ln = model_adapter.model.encoder.final_layer_norm
                if hasattr(ln, 'weight'):
                    ln.weight.data = ln.weight.data[:target_dim].contiguous()
                    if hasattr(ln, 'bias') and ln.bias is not None:
                        ln.bias.data = ln.bias.data[:target_dim].contiguous()
                    ln.normalized_shape = (target_dim,)
            
            # Adjust decoder cross-attention K/V to accept encoder output
            for layer_adapter in model_adapter.get_decoder_layers():
                layer = layer_adapter.layer
                cross_attn = layer.layer[1].EncDecAttention
                
                for param_name in ['k', 'v']:
                    W = getattr(cross_attn, param_name)
                    if W.weight.shape[1] != target_dim:
                        # Slice input dimension to match encoder output
                        W.weight.data = W.weight.data[:, :target_dim].contiguous()
                        W.in_features = target_dim
            
            logging.info(f"✓ Encoder-only slicing applied: encoder {target_dim} dims, decoder {orig_dim} dims")
        else:
            logging.warning("Could not apply encoder-only slicing - no encoder layers found")
            slice_rotated_model(model_adapter)
    else:
        # Standard full-model slicing
        slice_rotated_model(model_adapter)
    
    model = model_adapter.model

    # 4) enforce T5 invariant: lm_head tied to shared (FIXED ORDER)
    if hasattr(model, "shared") and hasattr(model, "lm_head"):
        shared_dim = int(model.shared.weight.shape[1])
        
        # Ensure lm_head matches shared dimension
        if model.lm_head.in_features != shared_dim:
            logging.warning(f"lm_head in_features ({model.lm_head.in_features}) != shared dim ({shared_dim}), fixing...")
            # Don't slice - just tie directly (checkpoint already has correct dims)
            model.lm_head.in_features = shared_dim
        
        # Tie weights (must be same object reference)
        model.lm_head.weight = model.shared.weight
        
        # Verify they're actually tied
        if model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr():
            raise RuntimeError("Failed to tie lm_head to shared embedding!")

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
    # Detect model dtype from first parameter
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    def _eye(n: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.eye(n, dtype=model_dtype, device=model_device))

    def _projection(in_dim: int, out_dim: int) -> torch.nn.Parameter:
        """Create a projection matrix for dimension mismatch in residual connections."""
        if in_dim == out_dim:
            return _eye(in_dim)
        # Create rectangular identity-like projection
        # For expansion (in < out): pad with zeros
        # For reduction (in > out): truncate
        proj = torch.zeros(in_dim, out_dim, dtype=model_dtype, device=model_device)
        min_dim = min(in_dim, out_dim)
        proj[:min_dim, :min_dim] = torch.eye(min_dim, dtype=model_dtype, device=model_device)
        return torch.nn.Parameter(proj)

    def _get_layer_dim(layer_idx: int, default_dim: int, dim_type: str = 'attention') -> int:
        """Get the dimension for a specific layer from slicing config.

        Args:
            layer_idx: Layer index
            default_dim: Fallback dimension
            dim_type: 'attention' for attention_input_dimensions, 'mlp' for mlp_output_dimensions
        """
        sc = model_adapter.slicing_conf

        # If const_dimension exists, use it for all layers
        if sc.const_dimension is not None:
            return int(sc.const_dimension)

        # Choose the right dimension dict based on type
        if dim_type == 'mlp':
            dim_dict = getattr(sc, 'mlp_output_dimensions', None)
        else:  # 'attention'
            dim_dict = getattr(sc, 'attention_input_dimensions', None)

        # For per-layer dimensions, look up the layer
        if dim_dict:
            # Try both string and int keys
            if str(layer_idx) in dim_dict:
                return int(dim_dict[str(layer_idx)])
            elif layer_idx in dim_dict:
                return int(dim_dict[layer_idx])

        # Fallback to default
        return int(default_dim)

    # Handle both full-model and encoder-only slicing
    enc_layers = model_adapter.get_encoder_layers() if hasattr(model_adapter, "get_encoder_layers") else model_adapter.get_layers()

    # Encoder shortcuts: use per-layer dimensions
    for i, la in enumerate(enc_layers):
        layer = la.layer
        # Attention shortcut uses attention input dimension
        attn_dim = _get_layer_dim(i, target_dim, dim_type='attention')
        layer.attn_shortcut_Q = _eye(attn_dim)

        # MLP shortcut: needs to handle dimension mismatch when MLP output != attention output
        if hasattr(layer, "mlp_shortcut_Q"):
            mlp_out_dim = _get_layer_dim(i, target_dim, dim_type='mlp')
            # Residual has attention output dimension, MLP has mlp_out_dim
            # Shortcut transforms residual (attn_dim) to match MLP output (mlp_out_dim)
            layer.mlp_shortcut_Q = _projection(attn_dim, mlp_out_dim)

    # Decoder shortcuts: detect actual dimension from weights
    if hasattr(model_adapter, "get_decoder_layers"):
        for la in model_adapter.get_decoder_layers():
            layer = la.layer
            
            # Infer decoder dimension from actual weight shapes
            # (For encoder-only slicing, decoder stays at orig_dim)
            dec_sa_inputs = la.get_attention_inputs()
            dec_dim = dec_sa_inputs[0].weight.shape[1] if dec_sa_inputs else orig_dim
            
            layer.attn_shortcut_Q = _eye(dec_dim)
            
            if hasattr(layer, "cross_attn_shortcut_Q"):
                layer.cross_attn_shortcut_Q = _eye(dec_dim)
            
            if hasattr(layer, "mlp_shortcut_Q"):
                dec_mlp_inputs = la.get_mlp_inputs()
                dec_mlp_dim = dec_mlp_inputs[0].weight.shape[1] if dec_mlp_inputs else dec_dim
                layer.mlp_shortcut_Q = _eye(dec_mlp_dim)
    
    # ========================================================================
    # NEW: ADD PROJECTION LAYER FOR ENCODER-ONLY SLICED T5 MODELS
    # ========================================================================
    
    # Detect if this is an encoder-only sliced model (T5/seq2seq)
    if hasattr(model, 'encoder') and hasattr(model, 'decoder') and hasattr(model, 'shared'):
        # Get dimensions
        embedding_dim = model.shared.weight.shape[1]
        
        # Get encoder input dimension from first encoder layer
        if hasattr(model_adapter, 'get_encoder_layers'):
            enc_layers_list = list(model_adapter.get_encoder_layers())
            if enc_layers_list:
                first_enc_layer = enc_layers_list[0]
                enc_attn_inputs = first_enc_layer.get_attention_inputs()
                if enc_attn_inputs:
                    encoder_input_dim = enc_attn_inputs[0].weight.shape[1]  # in_features
                    
                    # Check if projection is needed
                    needs_projection = (encoder_input_dim < embedding_dim)
                    
                    if needs_projection:
                        logging.info("="*70)
                        logging.info("ENCODER-ONLY SLICING DETECTED")
                        logging.info("="*70)
                        logging.info(f"Embedding dimension: {embedding_dim}")
                        logging.info(f"Encoder input dimension: {encoder_input_dim}")
                        logging.info(f"Adding projection layer: {embedding_dim} → {encoder_input_dim}")
                        
                        # Create projection layer (simple truncation)
                        projection = nn.Linear(embedding_dim, encoder_input_dim, bias=False)
                        with torch.no_grad():
                            # Initialize with identity truncation
                            projection.weight.data = torch.eye(encoder_input_dim, embedding_dim, dtype=torch.float32)
                        
                        # Store projection in model
                        model.encoder_projection = projection
                        
                        # CRITICAL: T5 encoder has its own embed_tokens that references model.shared
                        # We need to wrap it so that it applies projection automatically
                        class ProjectedEmbedding(nn.Module):
                            def __init__(self, original_embedding, projection):
                                super().__init__()
                                self.original_embedding = original_embedding
                                self.projection = projection
                                # Copy attributes for compatibility
                                self.num_embeddings = original_embedding.num_embeddings
                                self.embedding_dim = projection.out_features  # Projected dimension
                                self.padding_idx = original_embedding.padding_idx
                                
                                # Cache projected weight (don't recompute every time)
                                self._weight_cache = None
                                self._weight_dirty = True
                            
                            def forward(self, input_ids):
                                # Get full embeddings
                                embeds = self.original_embedding(input_ids)
                                # Apply projection
                                return self.projection(embeds)
                            
                            @property
                            def weight(self):
                                # Return projected weight for compatibility
                                # Cache it to avoid recomputation
                                if self._weight_dirty or self._weight_cache is None:
                                    with torch.no_grad():
                                        self._weight_cache = self.projection(self.original_embedding.weight.detach())
                                    self._weight_dirty = False
                                return self._weight_cache
                            
                            def _invalidate_cache(self):
                                self._weight_dirty = True
                        
                        # Replace encoder's embed_tokens with projected version
                        if hasattr(model.encoder, 'embed_tokens'):
                            model.encoder.embed_tokens = ProjectedEmbedding(model.shared, projection)
                            logging.info("✓ Replaced encoder.embed_tokens with projected version")
                        
                        # Also wrap encoder forward method as backup (for inputs_embeds path)
                        original_encoder_forward = model.encoder.forward
                        
                        def encoder_forward_with_projection(
                            input_ids=None,
                            attention_mask=None,
                            inputs_embeds=None,
                            head_mask=None,
                            output_attentions=None,
                            output_hidden_states=None,
                            return_dict=None,
                            **kwargs,  # Catch any additional kwargs
                        ):
                            """
                            Modified encoder forward that applies projection to embeddings.
                            
                            Note: If input_ids is provided, encoder.embed_tokens (which we replaced)
                            will handle the projection automatically. This wrapper is mainly for
                            the inputs_embeds path.
                            """
                            # If inputs_embeds is provided directly, we need to project it
                            if inputs_embeds is not None:
                                inputs_embeds = model.encoder_projection(inputs_embeds)
                            
                            # Call original encoder
                            # If input_ids is provided, encoder will use its embed_tokens (already projected)
                            # If inputs_embeds is provided, we just projected it above
                            return original_encoder_forward(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                inputs_embeds=inputs_embeds,
                                head_mask=head_mask,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                                **kwargs,
                            )
                        
                        # Replace encoder forward method
                        model.encoder.forward = encoder_forward_with_projection
                        
                        logging.info("✓ Projection layer created and encoder forward method wrapped")
                        logging.info("✓ Model can now perform forward passes")
                        logging.info("="*70)
    
    # ========================================================================
    # END OF PROJECTION LAYER ADDITION
    # ========================================================================
                
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
