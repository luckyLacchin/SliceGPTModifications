# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase

# Import Gemma 2 classes
from transformers.models.gemma.modeling_gemma import GemmaConfig, GemmaDecoderLayer, GemmaForCausalLM, GemmaRMSNorm

# Try to import Gemma 3 classes (available in transformers >= 4.50)
try:
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    # Try to import internal classes - may not be exported
    try:
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3TextDecoderLayer,
            Gemma3RMSNorm,
        )
    except (ImportError, AttributeError):
        # If internal classes not available, we'll dynamically get them from the model
        Gemma3TextDecoderLayer = None
        Gemma3RMSNorm = None
    HAS_GEMMA3 = True
    import logging
    logging.debug("Successfully imported Gemma 3 classes")
except ImportError as e:
    import logging
    logging.debug(f"Failed to import Gemma 3 classes: {e}")
    HAS_GEMMA3 = False
    # Fallback to Gemma 2 classes for type hints
    Gemma3TextConfig = GemmaConfig
    Gemma3TextDecoderLayer = GemmaDecoderLayer
    Gemma3ForCausalLM = GemmaForCausalLM
    Gemma3RMSNorm = GemmaRMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressedGemmaDecoderLayer(GemmaDecoderLayer):
    """
    This class simulates the GemmaDecoderLayer class from transformers
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py)
    but with the addition of shortcut_Q attributes. These attributes are used to rotate the residual tensors.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_value: tuple[Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: LongTensor | None = None,
        **kwargs,
    ) -> tuple:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Handle different return signatures (Gemma 2 vs Gemma 3)
        if isinstance(attn_output, tuple):
            if len(attn_output) == 3:
                hidden_states, self_attn_weights, present_key_value = attn_output
            elif len(attn_output) == 2:
                # Gemma 3 returns (hidden_states, cache) when use_cache=True
                # or (hidden_states, None) when use_cache=False
                hidden_states, present_key_value = attn_output
                self_attn_weights = None
            else:
                hidden_states = attn_output[0]
                self_attn_weights = None
                present_key_value = None
        else:
            hidden_states = attn_output
            self_attn_weights = None
            present_key_value = None

        if self.attn_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.mlp_shortcut_Q is not None:
            rotated_residual = matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        # Gemma 3 returns just the hidden_states tensor, not a tuple
        # This matches the original GemmaDecoderLayer behavior
        return hidden_states


# Will be created dynamically when needed
CompressedGemma3TextDecoderLayer = None


def create_compressed_gemma3_layer_class(base_layer_type):
    """
    Dynamically create a compressed Gemma3TextDecoderLayer class.
    This is needed because Gemma3TextDecoderLayer may not be directly importable.
    """

    class CompressedGemma3TextDecoderLayer(base_layer_type):
        """
        Compressed version of Gemma3TextDecoderLayer with shortcut_Q rotation support.
        This class extends Gemma3TextDecoderLayer to add rotation matrices for residual connections.l
        """

        def forward(
            self,
            hidden_states: Tensor,
            position_embeddings: Tensor | None = None,
            attention_mask: Tensor | None = None,
            position_ids: LongTensor | None = None,
            past_key_value: tuple[Tensor] | None = None,
            output_attentions: bool | None = False,
            use_cache: bool | None = False,
            cache_position: LongTensor | None = None,
            **kwargs,
        ) -> Tensor:
            """
            Forward pass with rotation support for SliceGPT.
            Note: Gemma 3 requires position_embeddings parameter.
            """
            import sys

            # DEBUG: Track what we receive at the start
            print(f"[FORWARD START] Received hidden_states.shape={hidden_states.shape}, ndim={hidden_states.ndim}", flush=True, file=sys.stderr)

            # CRITICAL FIX: Handle 0D tensors (scalars) by converting to 1D
            # This can happen when the model's forward loop incorrectly processes our 1D outputs
            if hidden_states.ndim == 0:
                # 0D scalar [] → 1D [1] by adding a dimension
                hidden_states = hidden_states.unsqueeze(0)
                print(f"[FORCE_3D] Converted 0D→1D: hidden_states.shape={hidden_states.shape}", flush=True, file=sys.stderr)

            # CRITICAL FIX: Force 3D structure to ensure consistent behavior
            # Gemma 3 supports 1D [hidden], 2D [seq, hidden], and 3D [batch, seq, hidden] inputs,
            # but position_embeddings are always generated as 3D [1, seq, head_dim].
            # To avoid shape mismatches during attention, we force all inputs to be 3D.
            # We track the original dimensionality to restore it at the end.
            original_ndim = hidden_states.ndim
            added_dims = False

            if hidden_states.ndim == 1:
                # 1D input [hidden] → add seq and batch dimensions
                # This represents a single position/token, so we also need to slice
                # attention_mask and position_embeddings to seq_len=1
                original_hidden_shape = hidden_states.shape
                hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)  # [hidden] → [1, 1, hidden]
                added_dims = True

                # CRITICAL: Adjust attention_mask to seq_len=1
                # Take only the last position: [batch, 1, 2048, 2048] → [1, 1, 1, 1]
                if attention_mask is not None and attention_mask.ndim == 4:
                    original_mask_shape = attention_mask.shape
                    # Slice to get only the last position for both dimensions
                    attention_mask = attention_mask[:1, :, -1:, -1:]
                    print(f"[FORCE_3D] Converted 1D→3D: hidden_states {original_hidden_shape}→{hidden_states.shape}, attention_mask {original_mask_shape}→{attention_mask.shape}", flush=True, file=sys.stderr)
                else:
                    print(f"[FORCE_3D] Converted 1D→3D: hidden_states {original_hidden_shape}→{hidden_states.shape}, attention_mask=None or wrong ndim", flush=True, file=sys.stderr)

                # Note: position_embeddings adjustment happens later (after it's computed)

            elif hidden_states.ndim == 2:
                # 2D input: Could be [seq, hidden] OR [1, hidden] (single token from previous 1D layer)
                original_hidden_shape = hidden_states.shape
                seq_len = hidden_states.shape[0]

                hidden_states = hidden_states.unsqueeze(0)  # [seq, hidden] → [1, seq, hidden]
                added_dims = True

                # CRITICAL: Also adjust attention_mask to match the new batch size AND seq_len
                # attention_mask typically has shape [batch, 1, seq, seq] or [batch, num_heads, seq, seq]
                # When we force hidden_states to batch=1, we need attention_mask to also be batch=1
                if attention_mask is not None and attention_mask.ndim == 4:
                    original_mask_shape = attention_mask.shape
                    # Take only the first element of the batch dimension
                    # [batch, ...] → [1, ...]
                    attention_mask = attention_mask[:1]

                    # CRITICAL: If seq_len=1, we need to slice attention_mask to [1, 1, 1, 1]
                    # This happens when 2D input [1, hidden] comes from a previous 1D layer output
                    if seq_len == 1:
                        attention_mask = attention_mask[:, :, -1:, -1:]
                        print(f"[FORCE_3D] Converted 2D→3D (seq_len=1): hidden_states {original_hidden_shape}→{hidden_states.shape}, attention_mask {original_mask_shape}→{attention_mask.shape}", flush=True, file=sys.stderr)
                    else:
                        print(f"[FORCE_3D] Converted 2D→3D: hidden_states {original_hidden_shape}→{hidden_states.shape}, attention_mask {original_mask_shape}→{attention_mask.shape}", flush=True, file=sys.stderr)
                else:
                    print(f"[FORCE_3D] Converted 2D→3D: hidden_states {original_hidden_shape}→{hidden_states.shape}, attention_mask=None or wrong ndim", flush=True, file=sys.stderr)

            residual = hidden_states
            print(f"[DEBUG] After adding batch dim (if needed): hidden_states.shape={hidden_states.shape}, residual.shape={residual.shape}", flush=True, file=sys.stderr)

            hidden_states = self.input_layernorm(hidden_states)
            print(f"[DEBUG] After input_layernorm: hidden_states.shape={hidden_states.shape}, ndim={hidden_states.ndim}", flush=True, file=sys.stderr)

            # Gemma 3 requires position_embeddings - if not provided, compute them
            # This happens when layers are called individually (e.g., during SliceGPT rotation)
            if position_embeddings is None:
                import logging

                # Ensure we have position_ids
                if position_ids is None:
                    batch_size, seq_length = hidden_states.shape[:2]
                    position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

                # In Gemma 3, rotary_emb is a standalone module, not inside self_attn
                # We need to access it from the model level, but when layers are called individually,
                # we don't have direct access. The solution is to use the attention module's
                # internal rotary embedding application.

                # Try to find rotary_emb in different locations
                rotary_emb = None
                if hasattr(self.self_attn, 'rotary_emb'):
                    rotary_emb = self.self_attn.rotary_emb
                elif hasattr(self, 'rotary_emb'):
                    rotary_emb = self.rotary_emb

                if rotary_emb is not None:
                    # Compute position embeddings
                    position_embeddings = rotary_emb(hidden_states, position_ids)
                    logging.info(f"Computed position_embeddings: {type(position_embeddings)}")
                else:
                    # Fallback: Create position embeddings manually using standard RoPE formula
                    # This is a workaround for when rotary_emb is not accessible
                    logging.info("rotary_emb not found, using fallback to pass None (attention will handle it)")
                    # For Gemma 3, if position_embeddings is None, the attention module
                    # should be able to compute it internally. However, this may not work
                    # during SliceGPT processing. We'll let it try and see the error.
                    pass

            # CRITICAL: If we converted 1D→3D or 2D(seq=1)→3D, adjust position_embeddings to match seq_len=1
            # This needs to happen AFTER position_embeddings is computed/provided
            if added_dims and (original_ndim == 1 or (original_ndim == 2 and hidden_states.shape[1] == 1)):
                if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)):
                    cos, sin = position_embeddings
                    original_pos_shape = cos.shape
                    # Take only the last position: [1, 2048, 256] → [1, 1, 256]
                    cos = cos[:, -1:, :]
                    sin = sin[:, -1:, :]
                    position_embeddings = (cos, sin)
                    print(f"[FORCE_3D] Adjusted position_embeddings for seq_len=1 (AFTER compute): {original_pos_shape}→{cos.shape}", flush=True, file=sys.stderr)

            # Self Attention - pass position_embeddings for Gemma 3
            # Since we've forced 3D structure, position_embeddings [1, seq, head_dim] will broadcast correctly

            # Debug logging to track tensor shapes
            print(f"[DEBUG] Before self_attn: hidden_states.shape={hidden_states.shape}, ndim={hidden_states.ndim}", flush=True, file=sys.stderr)
            if attention_mask is not None:
                print(f"[DEBUG] Before self_attn: attention_mask.shape={attention_mask.shape}", flush=True, file=sys.stderr)
            if position_embeddings is not None:
                print(f"[DEBUG] Before self_attn: position_embeddings shapes={[t.shape for t in position_embeddings] if isinstance(position_embeddings, (tuple, list)) else position_embeddings.shape}", flush=True, file=sys.stderr)

            try:
                attn_output = self.self_attn(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
                print(f"[DEBUG] After self_attn: attn_output type={type(attn_output)}, shape={attn_output.shape if hasattr(attn_output, 'shape') else 'N/A'}", flush=True, file=sys.stderr)
            except RuntimeError as e:
                # Re-raise with more context for debugging
                raise RuntimeError(
                    f"Error in CompressedGemma3TextDecoderLayer attention: {e}\n"
                    f"hidden_states.shape={hidden_states.shape}, "
                    f"position_embeddings={'None' if position_embeddings is None else f'tuple of shapes {[t.shape for t in position_embeddings]}'}"
                ) from e

            # Handle attention output (Gemma 3 returns just tensor)
            if isinstance(attn_output, tuple):
                hidden_states = attn_output[0]
            else:
                hidden_states = attn_output

            # Apply rotation to residual if available
            if hasattr(self, 'attn_shortcut_Q') and self.attn_shortcut_Q is not None:
                rotated_residual = matmul(residual, self.attn_shortcut_Q)
                hidden_states = rotated_residual + hidden_states
            else:
                hidden_states = residual + hidden_states

            # Fully Connected (MLP)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

            # Apply rotation to MLP residual if available
            if hasattr(self, 'mlp_shortcut_Q') and self.mlp_shortcut_Q is not None:
                rotated_residual = matmul(residual, self.mlp_shortcut_Q)
                hidden_states = rotated_residual + hidden_states
            else:
                hidden_states = residual + hidden_states

            # Restore original dimensionality by removing dimensions we added
            if added_dims:
                # Always remove the batch dimension we added, restore to original ndim
                if original_ndim == 1:
                    # Was 1D: [1, 1, hidden] → [hidden]
                    # CRITICAL: Cannot use squeeze as it may create 0D. Use explicit view/reshape.
                    # Get the hidden dimension size
                    hidden_size = hidden_states.shape[-1]
                    hidden_states = hidden_states.view(hidden_size)
                    print(f"[FORCE_3D] Restored 3D→1D: hidden_states.shape={hidden_states.shape}", flush=True, file=sys.stderr)
                elif original_ndim == 2:
                    # Was 2D: [1, seq, hidden] → [seq, hidden]
                    #  Use view to explicitly reshape
                    seq_len = hidden_states.shape[1]
                    hidden_size = hidden_states.shape[2]
                    hidden_states = hidden_states.view(seq_len, hidden_size)
                    print(f"[FORCE_3D] Restored 3D→2D: hidden_states.shape={hidden_states.shape}", flush=True, file=sys.stderr)
                # If original_ndim == 3, no change needed - already 3D

            # DEBUG: Track what we're returning
            print(f"[FORWARD END] Returning hidden_states.shape={hidden_states.shape}, ndim={hidden_states.ndim}", flush=True, file=sys.stderr)

            return hidden_states

    return CompressedGemma3TextDecoderLayer


class GemmaLayerAdapter(LayerAdapter):
    def __init__(self, layer: GemmaDecoderLayer) -> None:
        super().__init__()
        self._layer: GemmaDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj

    def slice_attention_output_projections(self, new_dimension: int) -> None:
        """
        Custom slicing for Gemma 3 attention outputs.

        For Gemma 3, we need to slice BOTH:
        1. q_proj, k_proj, v_proj OUTPUT dimensions (out_features)
        2. o_proj INPUT dimension (to match the sliced q/k/v outputs)

        Standard SliceGPT only slices o_proj OUTPUT dimension, which is not enough.
        """
        import torch
        import sys

        print(f"\n[SLICE_ATTN_PROJS] Slicing to new_dimension={new_dimension}", flush=True, file=sys.stderr)

        # Slice ONLY q_proj OUTPUT dimension (k/v are smaller in GQA/MQA)
        # For Gemma 3 with GQA, k_proj and v_proj have fewer heads than q_proj
        q_proj = self.layer.self_attn.q_proj
        old_out = q_proj.out_features
        old_shape = q_proj.weight.data.shape

        if old_out > new_dimension:
            q_proj.weight.data = q_proj.weight.data[:new_dimension, :]
            if q_proj.bias is not None:
                q_proj.bias.data = q_proj.bias.data[:new_dimension]
            q_proj.out_features = new_dimension
            print(f"[SLICE_ATTN_PROJS] q_proj: out_features {old_out}->{q_proj.out_features}, weight {old_shape}->{q_proj.weight.data.shape}", flush=True, file=sys.stderr)
        else:
            print(f"[SLICE_ATTN_PROJS] q_proj: SKIPPED (out_features={old_out} <= target={new_dimension})", flush=True, file=sys.stderr)

        # k_proj and v_proj stay at their original size (they use fewer heads)
        print(f"[SLICE_ATTN_PROJS] k_proj: PRESERVED (out_features={self.layer.self_attn.k_proj.out_features})", flush=True, file=sys.stderr)
        print(f"[SLICE_ATTN_PROJS] v_proj: PRESERVED (out_features={self.layer.self_attn.v_proj.out_features})", flush=True, file=sys.stderr)

        # o_proj input dimension needs to match q_proj + k_proj + v_proj (for concatenation)
        # Actually for GQA, o_proj input should match q_proj output only
        o_proj = self.layer.self_attn.o_proj
        old_in = o_proj.in_features
        old_shape = o_proj.weight.data.shape

        # o_proj.in_features should match q_proj.out_features
        if old_in > new_dimension and q_proj.out_features == new_dimension:
            o_proj.weight.data = o_proj.weight.data[:, :new_dimension]
            o_proj.in_features = new_dimension
            print(f"[SLICE_ATTN_PROJS] o_proj: in_features {old_in}->{o_proj.in_features}, weight {old_shape}->{o_proj.weight.data.shape}", flush=True, file=sys.stderr)
        else:
            print(f"[SLICE_ATTN_PROJS] o_proj: PRESERVED (in_features={old_in})", flush=True, file=sys.stderr)


class GemmaModelAdapter(ModelAdapter):
    def __init__(self, model: GemmaForCausalLM) -> None:
        super().__init__()
        self._model: GemmaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return GemmaConfig

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        return False

    @property
    def original_layer_type(self) -> type:
        return GemmaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        return GemmaRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return GemmaLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedGemmaDecoderLayer

    @property
    def use_cache(self) -> bool:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> FloatTensor:
        return self.model(input_ids=input_ids).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module:
        pre_head_layernorm = self.model.model.norm
        assert isinstance(pre_head_layernorm, self.original_layer_norm_type)
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Gemma models don't have a pad token by default
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            self.config.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("google/gemma"):
            return None

        # Check if this is a Gemma 3 model
        if HAS_GEMMA3 and "gemma-3" in model_name.lower():
            import logging
            logging.info(f"Loading Gemma 3 model: {model_name} with Gemma3ForCausalLM")
            model = Gemma3ForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
            )
            model.config.torch_dtype = dtype
            logging.info(f"Returning Gemma3ModelAdapter for {model_name}")
            return Gemma3ModelAdapter(model)
        else:
            # Gemma 2 or earlier
            model = GemmaForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
            )
            model.config.torch_dtype = dtype
            return GemmaModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str,
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> ModelAdapter | None:
        if not model_name.startswith("google/gemma"):
            return None

        class UninitializedGemmaForCausalLM(GemmaForCausalLM):
            def _init_weights(self, _) -> None:
                # Prevent weight initialization
                pass

        # Check if this is a Gemma 3 model
        if HAS_GEMMA3 and "gemma-3" in model_name.lower():
            class UninitializedGemma3ForCausalLM(Gemma3ForCausalLM):
                def _init_weights(self, _) -> None:
                    pass

            config = Gemma3TextConfig.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
            )
            model = UninitializedGemma3ForCausalLM(config)
            model = model.to(dtype=dtype)
            return Gemma3ModelAdapter(model)
        else:
            config = GemmaConfig.from_pretrained(
                model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
            )
            model = UninitializedGemmaForCausalLM(config)
            model = model.to(dtype=dtype)
            return GemmaModelAdapter(model)


class Gemma3ModelAdapter(GemmaModelAdapter):
    """
    Adapter for Gemma 3 models (google/gemma-3-*).
    Inherits most functionality from GemmaModelAdapter but uses Gemma 3-specific classes.
    """

    def __init__(self, model: Gemma3ForCausalLM) -> None:
        # Call ModelAdapter.__init__ directly to avoid GemmaModelAdapter's type annotation
        ModelAdapter.__init__(self)
        self._model: Gemma3ForCausalLM = model

        # Dynamically get layer types if not available via imports
        global Gemma3TextDecoderLayer, Gemma3RMSNorm, CompressedGemma3TextDecoderLayer
        if Gemma3TextDecoderLayer is None and len(model.model.layers) > 0:
            Gemma3TextDecoderLayer = type(model.model.layers[0])
        if Gemma3RMSNorm is None:
            Gemma3RMSNorm = type(model.model.norm)

        # Create compressed layer class dynamically if needed
        if CompressedGemma3TextDecoderLayer is None and Gemma3TextDecoderLayer is not None:
            CompressedGemma3TextDecoderLayer = create_compressed_gemma3_layer_class(Gemma3TextDecoderLayer)

        # Store the compressed layer type for this instance
        self._compressed_layer_type = CompressedGemma3TextDecoderLayer

    @property
    def config_type(self) -> type:
        return Gemma3TextConfig

    @property
    def original_layer_type(self) -> type:
        if Gemma3TextDecoderLayer is not None:
            return Gemma3TextDecoderLayer
        # Fallback: get from actual model
        return type(self.model.model.layers[0])

    @property
    def original_layer_norm_type(self) -> type:
        if Gemma3RMSNorm is not None:
            return Gemma3RMSNorm
        # Fallback: get from actual model
        return type(self.model.model.norm)

    @property
    def compressed_layer_type(self) -> type:
        if hasattr(self, '_compressed_layer_type') and self._compressed_layer_type is not None:
            return self._compressed_layer_type
        # Fallback
        return CompressedGemma3TextDecoderLayer if CompressedGemma3TextDecoderLayer is not None else CompressedGemmaDecoderLayer

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        import sys
        compressed_layer = self.compressed_layer_type(self.config, layer_idx).to(self.config.torch_dtype)
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)

        # Gemma 3 specific: Inject rotary_emb reference from the model
        # In Gemma 3, rotary_emb is a shared model-level module, not per-layer
        if hasattr(self.model.model, 'rotary_emb'):
            compressed_layer.rotary_emb = self.model.model.rotary_emb
            import logging
            logging.info(f"Injected rotary_emb reference into compressed layer {layer_idx}")

        # CRITICAL FIX: Create a hook to update head_dim after slicing
        # The hook will be called when q_proj weights are sliced
        def update_head_dim_after_slicing():
            """Update attention head_dim to match sliced q_proj dimensions"""
            import sys
            print(f"\n[UPDATE_HEAD_DIM_HOOK] Called for layer", flush=True, file=sys.stderr)
            if hasattr(compressed_layer, 'self_attn') and hasattr(compressed_layer.self_attn, 'q_proj'):
                q_proj = compressed_layer.self_attn.q_proj
                if hasattr(q_proj, 'out_features'):
                    # CRITICAL: Get num_heads from the attention module itself, NOT from config
                    # The attention module stores num_heads or num_attention_heads
                    num_heads = getattr(compressed_layer.self_attn, 'num_heads', None)
                    if num_heads is None:
                        num_heads = getattr(compressed_layer.self_attn, 'num_attention_heads', None)
                    if num_heads is None:
                        # Fallback to config
                        num_heads = getattr(self.config, 'num_attention_heads', None) or getattr(self.config, 'num_heads', None)
                    if num_heads is None:
                        num_heads = 4  # Default for Gemma 3 270M

                    new_head_dim = q_proj.out_features // num_heads
                    old_head_dim = getattr(compressed_layer.self_attn, 'head_dim', None)

                    print(f"[UPDATE_HEAD_DIM] q_proj.out_features={q_proj.out_features}, num_heads={num_heads}, old_head_dim={old_head_dim}, new_head_dim={new_head_dim}", flush=True, file=sys.stderr)

                    # For Gemma 3 GQA: DON'T update head_dim or k_norm!
                    # head_dim is used for k/v which have different (smaller) dimensions than q
                    # Only update q_norm to match the new query head dimension
                    print(f"[UPDATE_HEAD_DIM] Gemma 3 GQA detected - preserving head_dim={old_head_dim} for k/v, only updating q_norm", flush=True, file=sys.stderr)

                    if hasattr(compressed_layer.self_attn, 'q_norm') and hasattr(compressed_layer.self_attn.q_norm, 'weight'):
                        old_weight = compressed_layer.self_attn.q_norm.weight
                        if old_weight.shape[0] != new_head_dim:
                            compressed_layer.self_attn.q_norm.weight = torch.nn.Parameter(old_weight[:new_head_dim])
                            print(f"[UPDATE_HEAD_DIM] Updated q_norm.weight from shape {old_weight.shape} to {compressed_layer.self_attn.q_norm.weight.shape}", flush=True, file=sys.stderr)

                    # k_norm stays at original size since k has different head_dim than q in GQA
                    print(f"[UPDATE_HEAD_DIM] k_norm preserved at original size for GQA", flush=True, file=sys.stderr)
                else:
                    print(f"[UPDATE_HEAD_DIM] ERROR: q_proj has no out_features", flush=True, file=sys.stderr)
            else:
                print(f"[UPDATE_HEAD_DIM] ERROR: No self_attn or q_proj found", flush=True, file=sys.stderr)

        # Store the hook function so it can be called after slicing
        compressed_layer._update_head_dim_hook = update_head_dim_after_slicing

        return compressed_layer
