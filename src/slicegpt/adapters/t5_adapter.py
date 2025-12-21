# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# T5 / FLAN-T5 adapter for SliceGPT (seq2seq).
# Based on HF transformers T5 implementation.

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Module

from transformers import PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5Config,
    T5ForConditionalGeneration,
    T5LayerNorm,
)

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


# ----------------------------
# Compressed layer
# ----------------------------

class CompressedT5Block(T5Block):
    """
    T5Block with shortcut rotations for SliceGPT slicing.

    We replicate the residual-add logic of T5LayerSelfAttention / T5LayerCrossAttention / T5LayerFF
    so that we can rotate the residual branch before addition.

    Buffers registered by ModelAdapter.convert_layer_to_compressed_and_register_buffers():
      - attn_shortcut_Q
      - cross_attn_shortcut_Q (decoder only)
      - mlp_shortcut_Q
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        cache_position=None,
    ):
        # ---- Self-attention sublayer (layer[0]) ----
        # Original T5LayerSelfAttention:
        # normed = ln(hidden_states)
        # attn_out = SelfAttention(normed, ...)
        # hidden_states = hidden_states + dropout(attn_out[0])
        # outputs = (hidden_states,) + attn_out[1:]
        sa = self.layer[0]
        residual = hidden_states

        normed = sa.layer_norm(hidden_states)
        attn_out = sa.SelfAttention(
            normed,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        attn_hidden = sa.dropout(attn_out[0])

        if getattr(self, "attn_shortcut_Q", None) is not None:
            residual = torch.matmul(residual, self.attn_shortcut_Q)
        hidden_states = residual + attn_hidden

        attention_outputs = attn_out[1:]  # position bias, attn weights, (maybe present_key_value)

        # clamp inf values to enable fp16 training (copied from HF)
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # ---- Cross-attention (decoder only, layer[1]) ----
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            ca = self.layer[1]
            residual = hidden_states

            normed = ca.layer_norm(hidden_states)
            cross_out = ca.EncDecAttention(
                normed,
                mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                position_bias=encoder_decoder_position_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                query_length=cache_position[-1] + 1 if cache_position is not None else None,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            cross_hidden = ca.dropout(cross_out[0])

            if getattr(self, "cross_attn_shortcut_Q", None) is not None:
                residual = torch.matmul(residual, self.cross_attn_shortcut_Q)
            hidden_states = residual + cross_hidden

            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            attention_outputs = attention_outputs + cross_out[1:]

        # ---- Feed-forward (layer[-1]) ----
        ff = self.layer[-1]
        residual = hidden_states

        ff_in = ff.layer_norm(hidden_states)
        ff_out = ff.DenseReluDense(ff_in)
        ff_out = ff.dropout(ff_out)

        if getattr(self, "mlp_shortcut_Q", None) is not None:
            residual = torch.matmul(residual, self.mlp_shortcut_Q)
        hidden_states = residual + ff_out

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        return outputs + attention_outputs


# ----------------------------
# Layer adapters
# ----------------------------

class T5BlockLayerAdapter(LayerAdapter):
    """
    Adapter for a T5Block (encoder or decoder).
    For decoder cross-attn specific accessors, use T5DecoderBlockLayerAdapter.
    """

    def __init__(self, layer: T5Block) -> None:
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
        # Self-attn LN
        return self._layer.layer[0].layer_norm

    def get_second_layernorm(self) -> Module:
        # FFN LN
        return self._layer.layer[-1].layer_norm

    def get_attention_inputs(self) -> Sequence[Linear]:
        attn = self._layer.layer[0].SelfAttention
        return [attn.q, attn.k, attn.v]

    def get_attention_output(self) -> Linear:
        attn = self._layer.layer[0].SelfAttention
        return attn.o

    def get_mlp_inputs(self) -> Sequence[Linear]:
        # DenseReluDense can be gated or not.
        drd = self._layer.layer[-1].DenseReluDense
        if hasattr(drd, "wi"):
            return [drd.wi]
        # gated
        return [drd.wi_0, drd.wi_1]

    def get_mlp_output(self) -> Linear:
        drd = self._layer.layer[-1].DenseReluDense
        return drd.wo


class T5DecoderBlockLayerAdapter(T5BlockLayerAdapter):
    """
    Decoder T5Block adapter with cross-attention accessors.
    """

    def get_cross_attention_layernorm(self) -> Module:
        if not self._layer.is_decoder:
            raise RuntimeError("Cross-attention LN requested on a non-decoder T5Block")
        return self._layer.layer[1].layer_norm

    def get_cross_attention_q_input(self) -> Linear:
        if not self._layer.is_decoder:
            raise RuntimeError("Cross-attention q requested on a non-decoder T5Block")
        ca = self._layer.layer[1].EncDecAttention
        return ca.q

    def get_cross_attention_kv_inputs(self) -> Sequence[Linear]:
        if not self._layer.is_decoder:
            raise RuntimeError("Cross-attention k/v requested on a non-decoder T5Block")
        ca = self._layer.layer[1].EncDecAttention
        return [ca.k, ca.v]

    def get_cross_attention_output(self) -> Linear:
        if not self._layer.is_decoder:
            raise RuntimeError("Cross-attention o requested on a non-decoder T5Block")
        ca = self._layer.layer[1].EncDecAttention
        return ca.o


# ----------------------------
# Model adapter
# ----------------------------

class T5ModelAdapter(ModelAdapter):
    """
    ModelAdapter for T5 / FLAN-T5 (seq2seq).
    """

    def __init__(self, model: T5ForConditionalGeneration) -> None:
        super().__init__()
        self._model = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return T5Config

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        # T5 configs typically use n_positions; fallback to max_length if present.
        return int(getattr(self.config, "n_positions", getattr(self.config, "max_length", 512)))

    @property
    def hidden_size(self) -> int:
        return int(self.config.d_model)

    @property
    def should_bake_mean_into_linear(self) -> bool:
        # T5LayerNorm is RMSNorm-like (no mean subtraction)
        return False

    @property
    def original_layer_type(self) -> type:
        return T5Block

    @property
    def original_layer_norm_type(self) -> type:
        return T5LayerNorm

    @property
    def layer_adapter_type(self) -> type:
        # For legacy calls (single stack), treat encoder as default.
        return T5BlockLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedT5Block

    @property
    def use_cache(self) -> bool:
        return bool(getattr(self.config, "use_cache", False))

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor) -> torch.FloatTensor:
        """
        For seq2seq, we need the decoder to run. The simplest deterministic way is to provide labels.
        This is only used for slicing calibration signal extraction in some parts; we keep it minimal.
        """
        attention_mask = torch.ones_like(input_ids)
        out = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return out.logits

    # ---- seq2seq-specific accessors used by your new catcher code ----

    def get_encoder_layers(self) -> list[LayerAdapter]:
        return [T5BlockLayerAdapter(layer) for layer in self._model.encoder.block]

    def get_decoder_layers(self) -> list[LayerAdapter]:
        return [T5DecoderBlockLayerAdapter(layer) for layer in self._model.decoder.block]

    # For older SliceGPT code that expects a single list of layers:
    def get_layers(self) -> list[LayerAdapter]:
        return self.get_encoder_layers()

    def get_raw_layer_at(self, index: int) -> Module:
        # legacy path: encoder
        return self._model.encoder.block[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        # legacy path: encoder
        self._model.encoder.block[index] = new_layer

    def get_raw_encoder_layer_at(self, index: int) -> Module:
        return self._model.encoder.block[index]

    def set_raw_encoder_layer_at(self, index: int, new_layer: Module) -> None:
        self._model.encoder.block[index] = new_layer

    def get_raw_decoder_layer_at(self, index: int) -> Module:
        return self._model.decoder.block[index]

    def set_raw_decoder_layer_at(self, index: int, new_layer: Module) -> None:
        self._model.decoder.block[index] = new_layer

    # ---- compression layer replacement ----

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        if not isinstance(layer, T5Block):
            raise TypeError(f"Expected T5Block, got {type(layer)}")

        # T5Block signature: (config, has_relative_attention_bias=False, layer_idx=None)
        # The first block in each stack may have relative attention bias.
        # We preserve this by reading whether the original block's self-attn has it.
        has_rpb = bool(getattr(layer.layer[0].SelfAttention, "has_relative_attention_bias", False))

        compressed = self.compressed_layer_type(self.config, has_relative_attention_bias=has_rpb, layer_idx=layer_idx)
        compressed.load_state_dict(layer.state_dict(), strict=True)

        # preserve decoder flag
        compressed.is_decoder = layer.is_decoder
        return compressed

    # ---- embeddings / head ----
    
    def convert_layer_to_compressed_and_register_buffers(
        self,
        layer: Module,
        layer_idx: int | None,
    ) -> Module:
        compressed = super().convert_layer_to_compressed_and_register_buffers(layer, layer_idx)

        # T5 decoder has cross-attention → needs its own shortcut rotation
        # Encoder blocks will simply keep this as None
        compressed.register_parameter("cross_attn_shortcut_Q", None)

        return compressed


    def get_embeddings(self) -> list[Module]:
        # Shared embedding matrix
        return [self._model.shared]

    def get_pre_head_layernorm(self) -> Module:
        # Decoder final layer norm exists in HF T5
        ln = self._model.decoder.final_layer_norm
        assert isinstance(ln, self.original_layer_norm_type)
        return ln

    def get_lm_head(self) -> Linear:
        return self._model.lm_head

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Ensure pad token exists (T5 typically has it, but safe)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            self.config.pad_token_id = tokenizer.pad_token_id

    # ---- model loading ----

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
        # Accept FLAN-T5 family (and optionally plain T5)
        if not (model_name.startswith("google/flan-t5") or model_name.startswith("t5-") or "flan-t5" in model_name):
            return None

        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
        )
        model.config.torch_dtype = dtype
        return cls(model)

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
        if not (model_name.startswith("google/flan-t5") or model_name.startswith("t5-") or "flan-t5" in model_name):
            return None

        class UninitializedT5ForConditionalGeneration(T5ForConditionalGeneration):
            def _init_weights(self, _module):
                # Prevent weight initialization
                pass

        config = T5Config.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
        )
        model = UninitializedT5ForConditionalGeneration(config).to(dtype=dtype)
        model.config.torch_dtype = dtype
        return cls(model)
