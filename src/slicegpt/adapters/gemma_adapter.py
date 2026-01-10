# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# This file contains derivations from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py
# and https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
# Copyright 2024 Google Inc. HuggingFace Inc. team. All Rights Reserved.
# https://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import FloatTensor, LongTensor, Tensor, matmul
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase, AutoConfig, AutoModelForCausalLM

from slicegpt.model_adapter import LayerAdapter, ModelAdapter

# Try importing Gemma classes - Gemma 1 and Gemma 3 share similar architecture
try:
    from transformers.models.gemma.modeling_gemma import GemmaConfig, GemmaDecoderLayer, GemmaForCausalLM, GemmaRMSNorm
    HAS_GEMMA = True
except ImportError:
    HAS_GEMMA = False

try:
    from transformers.models.gemma2.modeling_gemma2 import Gemma2Config, Gemma2DecoderLayer, Gemma2ForCausalLM, Gemma2RMSNorm
    HAS_GEMMA2 = True
except ImportError:
    HAS_GEMMA2 = False

try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3Config, Gemma3DecoderLayer, Gemma3ForCausalLM, Gemma3RMSNorm
    HAS_GEMMA3 = True
except ImportError:
    HAS_GEMMA3 = False


# Compressed Gemma (v1/v3) Decoder Layer
if HAS_GEMMA:
    class CompressedGemmaDecoderLayer(GemmaDecoderLayer):
        """
        Compressed Gemma decoder layer with shortcut_Q support.
        Works for Gemma 1 and Gemma 3 models.
        """

        def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor | None = None,
            position_ids: LongTensor | None = None,
            past_key_value: tuple[Tensor] | None = None,
            output_attentions: bool | None = False,
            use_cache: bool | None = False,
            cache_position: Tensor | None = None,
            **kwargs,
        ) -> tuple:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

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

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs


# Compressed Gemma3 Decoder Layer
if HAS_GEMMA3:
    class CompressedGemma3DecoderLayer(Gemma3DecoderLayer):
        """
        Compressed Gemma3 decoder layer with shortcut_Q support.
        Implements the slicing logic with proper residual connection handling.
        """

        def forward(
            self,
            hidden_states: Tensor,
            position_embeddings_global: Tensor,
            position_embeddings_local: Tensor,
            attention_mask: Tensor | None = None,
            position_ids: LongTensor | None = None,
            past_key_value: tuple[Tensor] | None = None,
            output_attentions: bool | None = False,
            use_cache: bool | None = False,
            cache_position: Tensor | None = None,
            **kwargs,
        ) -> tuple:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            if self.self_attn.is_sliding:
                position_embeddings = position_embeddings_local
            else:
                position_embeddings = position_embeddings_global

            # Self Attention
            attn_output = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            # Handle variable return values from attention
            # When output_attentions=False: (hidden_states, past_key_value)
            # When output_attentions=True: (hidden_states, attn_weights, past_key_value)
            hidden_states = attn_output[0]
            if output_attentions:
                self_attn_weights = attn_output[1]
                present_key_value = attn_output[2] if use_cache else None
            else:
                self_attn_weights = None
                present_key_value = attn_output[1] if use_cache else None

            # Apply shortcut to residual if slicing has occurred
            if hasattr(self, 'attn_shortcut_Q') and self.attn_shortcut_Q is not None:
                rotated_residual = matmul(residual, self.attn_shortcut_Q)
                hidden_states = rotated_residual + hidden_states
            else:
                hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)

            # Apply shortcut to residual if slicing has occurred
            if hasattr(self, 'mlp_shortcut_Q') and self.mlp_shortcut_Q is not None:
                rotated_residual = matmul(residual, self.mlp_shortcut_Q)
                hidden_states = rotated_residual + hidden_states
            else:
                hidden_states = residual + hidden_states

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs


# Compressed Gemma2 Decoder Layer
if HAS_GEMMA2:
    class CompressedGemma2DecoderLayer(Gemma2DecoderLayer):
        """
        Compressed Gemma2 decoder layer with shortcut_Q support.
        """

        def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor | None = None,
            position_ids: LongTensor | None = None,
            past_key_value: tuple[Tensor] | None = None,
            output_attentions: bool | None = False,
            use_cache: bool | None = False,
            cache_position: Tensor | None = None,
            **kwargs,
        ) -> tuple:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

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

            # Gemma2 uses post_feedforward_layernorm
            if hasattr(self, 'post_feedforward_layernorm') and self.post_feedforward_layernorm is not None:
                hidden_states = self.post_feedforward_layernorm(hidden_states)

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs


# Layer Adapter for Gemma (v1/v3)
class GemmaLayerAdapter(LayerAdapter):
    def __init__(self, layer) -> None:
        super().__init__()
        self._layer = layer

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

    @property
    def has_cross_attention(self) -> bool:
        return False

    def get_cross_attention_layernorm(self) -> Module:
        raise NotImplementedError("No cross-attention")

    def get_cross_attention_q_input(self) -> Linear:
        raise NotImplementedError("No cross-attention")

    def get_cross_attention_kv_inputs(self) -> list[Linear]:
        raise NotImplementedError("No cross-attention")

    def get_cross_attention_output(self) -> Linear:
        raise NotImplementedError("No cross-attention")


# Universal Gemma Model Adapter (works for Gemma 1, 2, and 3)
class GemmaModelAdapter(ModelAdapter):
    def __init__(self, model) -> None:
        super().__init__()
        self._model = model

        # Detect model type
        self._is_gemma3 = "Gemma3" in type(model).__name__
        self._is_gemma2 = "Gemma2" in type(model).__name__

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        if self._is_gemma3 and HAS_GEMMA3:
            return Gemma3Config
        elif self._is_gemma2 and HAS_GEMMA2:
            return Gemma2Config
        elif HAS_GEMMA:
            return GemmaConfig
        return type(self._model.config)

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
        if self._is_gemma3 and HAS_GEMMA3:
            return Gemma3DecoderLayer
        elif self._is_gemma2 and HAS_GEMMA2:
            return Gemma2DecoderLayer
        elif HAS_GEMMA:
            return GemmaDecoderLayer
        # Fallback to detecting from model structure
        return type(self._model.model.layers[0])

    @property
    def original_layer_norm_type(self) -> type:
        if self._is_gemma3 and HAS_GEMMA3:
            return Gemma3RMSNorm
        elif self._is_gemma2 and HAS_GEMMA2:
            return Gemma2RMSNorm
        elif HAS_GEMMA:
            return GemmaRMSNorm
        # Fallback
        return type(self._model.model.norm)

    @property
    def layer_adapter_type(self) -> type:
        return GemmaLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        if self._is_gemma3 and HAS_GEMMA3:
            return CompressedGemma3DecoderLayer
        elif self._is_gemma2 and HAS_GEMMA2:
            return CompressedGemma2DecoderLayer
        elif HAS_GEMMA:
            return CompressedGemmaDecoderLayer
        raise RuntimeError("No Gemma classes available")

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
        return pre_head_layernorm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Gemma models typically have pad tokens, but ensure it's set
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

        # Use AutoModel to automatically detect and load the correct Gemma variant
        model = AutoModelForCausalLM.from_pretrained(
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

        # Load config to determine model type
        config = AutoConfig.from_pretrained(
            model_path, torch_dtype=dtype, token=token, local_files_only=local_files_only
        )

        # Determine which model class to use
        arch_name = config.architectures[0] if hasattr(config, 'architectures') else ""
        is_gemma3 = "Gemma3" in arch_name
        is_gemma2 = "Gemma2" in arch_name

        if is_gemma3 and HAS_GEMMA3:
            class UninitializedGemma3ForCausalLM(Gemma3ForCausalLM):
                def _init_weights(self, _) -> None:
                    pass

            model = UninitializedGemma3ForCausalLM(config)
        elif is_gemma2 and HAS_GEMMA2:
            class UninitializedGemma2ForCausalLM(Gemma2ForCausalLM):
                def _init_weights(self, _) -> None:
                    pass

            model = UninitializedGemma2ForCausalLM(config)
        elif HAS_GEMMA:
            class UninitializedGemmaForCausalLM(GemmaForCausalLM):
                def _init_weights(self, _) -> None:
                    pass

            model = UninitializedGemmaForCausalLM(config)
        else:
            # Fallback to AutoModel
            model = AutoModelForCausalLM.from_config(config)

        model = model.to(dtype=dtype)

        return GemmaModelAdapter(model)


# Backward compatibility aliases
Gemma2ModelAdapter = GemmaModelAdapter
Gemma2LayerAdapter = GemmaLayerAdapter