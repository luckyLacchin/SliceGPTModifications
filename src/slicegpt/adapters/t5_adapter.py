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
from typing import Sequence, Any
from slicegpt.model_adapter import LayerAdapter, ModelAdapter
import inspect
import copy


# ============================================================
# Helpers: call modules with only supported kwargs (HF-version-proof)
# ============================================================

def _filter_kwargs_for_forward(mod: Module, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(mod.forward)
    allowed = sig.parameters.keys()
    return {k: v for k, v in kwargs.items() if k in allowed}

def _call_attn(attn_mod: Module, first_arg: Tensor, /, **kwargs):
    """
    Call T5Attention (or similar) robustly across Transformers versions.
    Uses positional first arg, filters kwargs to match attn_mod.forward signature.
    """
    filtered = _filter_kwargs_for_forward(attn_mod, kwargs)
    return attn_mod(first_arg, **filtered)


# ----------------------------
# Compressed layer
# ----------------------------

class CompressedT5Block(T5Block):
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        past_key_values=None,   # backward compat
        use_cache: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,  # ignored; T5Block returns tuples
        cache_position=None,
        **kwargs,               # swallow extra HF kwargs safely
    ):
        # ---------------------------
        # Helpers
        # ---------------------------
        def _bias_tensor(b):
            # Some wrappers accidentally keep bias as tuple; HF expects Tensor or None
            if isinstance(b, tuple):
                return b[0]
            return b

        def _maybe_clamp_fp16(x):
            if x is not None and x.dtype == torch.float16:
                # Prevent infs in fp16 (mimics HF behavior in some branches)
                finfo = torch.finfo(x.dtype)
                clamp_value = finfo.max
                if torch.isinf(x).any():
                    clamp_value = finfo.max - 1000
                return torch.clamp(x, min=-clamp_value, max=clamp_value)
            return x

        def _apply_shortcut(residual, Q):
            """
            Apply learned shortcut projection Q to residual if present.
            Q is expected as [in_dim, out_dim] but we handle transposed too.
            """
            if Q is None:
                return residual
            Qd = Q.to(device=residual.device, dtype=residual.dtype)

            # Standard: residual[..., in] @ Q[in,out] -> residual[..., out]
            if residual.shape[-1] == Qd.shape[0]:
                return residual @ Qd

            # If stored transposed: Q[out,in]
            if residual.shape[-1] == Qd.shape[1]:
                return residual @ Qd.transpose(0, 1)

            # If dims don’t match, just return unchanged (safer than crashing mid-slice)
            return residual

        def _match_last_dim(x, target_dim: int):
            """Force x last-dim to target_dim by slice/pad (safest during mid-slice)."""
            if x.shape[-1] == target_dim:
                return x
            if x.shape[-1] > target_dim:
                return x[..., :target_dim]
            pad = target_dim - x.shape[-1]
            return torch.nn.functional.pad(x, (0, pad))
        
        def _flatten_present_kv(self_kv, cross_kv):
            """
            Convert:
            self_kv  = (k, v) or None
            cross_kv = (k, v) or None
            into HF T5 expected:
            (self_k, self_v, cross_k, cross_v)
            """
            # Already flattened in some HF versions
            if isinstance(self_kv, (tuple, list)) and len(self_kv) == 4:
                return tuple(self_kv)

            if self_kv is None:
                sk = sv = None
            else:
                sk, sv = self_kv

            if cross_kv is None:
                ck = cv = None
            else:
                ck, cv = cross_kv

            return (sk, sv, ck, cv)

        # HF has used both names across versions; pick whichever is provided.
        pkv = past_key_value if past_key_value is not None else past_key_values

        # Decoder pkv layout in HF T5 can be either:
        #  - flat: (self_k, self_v, cross_k, cross_v)   [HF generation format]
        #  - nested: ((self_k, self_v), (cross_k, cross_v))  [some custom wrappers]
        #  - self-only: (self_k, self_v)
        self_pkv = None
        cross_pkv = None

        if self.is_decoder and pkv is not None and isinstance(pkv, (tuple, list)):
            if len(pkv) == 4:
                # HF standard for T5 generation
                self_pkv = (pkv[0], pkv[1])
                cross_pkv = (pkv[2], pkv[3])
            elif len(pkv) == 2:
                # Could be nested ((k,v),(k,v)) OR self-only (k,v)
                a, b = pkv[0], pkv[1]
                if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == 2 and len(b) == 2:
                    # nested format
                    self_pkv, cross_pkv = a, b
                else:
                    # self-only format
                    self_pkv = (a, b)
                    cross_pkv = None
        else:
            # encoder or no cache
            self_pkv = pkv
            cross_pkv = None


        # Sanitize biases
        position_bias = _bias_tensor(position_bias)
        encoder_decoder_position_bias = _bias_tensor(encoder_decoder_position_bias)

        # We'll collect these explicitly and then pack in correct order
        self_present_kv = None
        self_pos_bias = None
        self_attn_w = None

        cross_present_kv = None
        cross_pos_bias = None
        cross_attn_w = None

        # ---------------------------
        # 1) Self-attention (layer[0])
        # ---------------------------
        sa = self.layer[0]
        residual = hidden_states

        normed = sa.layer_norm(hidden_states)
        attn_out = _call_attn(
            sa.SelfAttention,
            normed,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_pkv,         # IMPORTANT: self pkv only
            use_cache=use_cache,
            output_attentions=output_attentions,
            layer_head_mask=layer_head_mask,
            cache_position=cache_position,
        )

        attn_hidden = sa.dropout(attn_out[0])
        residual = _apply_shortcut(residual, getattr(self, "attn_shortcut_Q", None))
        residual = _match_last_dim(residual, attn_hidden.shape[-1])
        hidden_states = residual + attn_hidden
        hidden_states = _maybe_clamp_fp16(hidden_states)

        # Unpack self-attn outputs (HF T5Attention):
        # (attn_output, present_kv, position_bias, attn_weights?) depending on flags
        if len(attn_out) > 1:
            self_present_kv = attn_out[1]
        if len(attn_out) > 2:
            self_pos_bias = _bias_tensor(attn_out[2])
        if output_attentions and len(attn_out) > 3:
            self_attn_w = attn_out[3]

        # Keep updated position_bias for next layer if provided
        if self_pos_bias is not None:
            position_bias = self_pos_bias

        # ---------------------------
        # 2) Cross-attention (decoder only, layer[1])
        # ---------------------------
        do_cross_attention = self.is_decoder and (encoder_hidden_states is not None)
        if do_cross_attention:
            ca = self.layer[1]
            residual = hidden_states

            normed = ca.layer_norm(hidden_states)
            cross_out = _call_attn(
                ca.EncDecAttention,
                normed,
                mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                position_bias=encoder_decoder_position_bias,
                past_key_value=cross_pkv,      # IMPORTANT: cross pkv only
                use_cache=use_cache,
                output_attentions=output_attentions,
                layer_head_mask=cross_attn_layer_head_mask,
                cache_position=cache_position,
                # Some HF versions accept query_length; harmless if filtered out
                query_length=(cache_position[-1] + 1) if cache_position is not None else None,
            )

            cross_hidden = ca.dropout(cross_out[0])

            residual = _apply_shortcut(residual, getattr(self, "cross_attn_shortcut_Q", None))
            residual = _match_last_dim(residual, cross_hidden.shape[-1])
            hidden_states = residual + cross_hidden
            hidden_states = _maybe_clamp_fp16(hidden_states)

            # Unpack cross-attn outputs
            if len(cross_out) > 1:
                cross_present_kv = cross_out[1]
            if len(cross_out) > 2:
                cross_pos_bias = _bias_tensor(cross_out[2])
            if output_attentions and len(cross_out) > 3:
                cross_attn_w = cross_out[3]

            # Keep updated encoder_decoder_position_bias for next layer
            if cross_pos_bias is not None:
                encoder_decoder_position_bias = cross_pos_bias

        # ---------------------------
        # 3) Feed-forward (layer[-1])
        # ---------------------------
        ff = self.layer[-1]
        residual = hidden_states

        ff_in = ff.layer_norm(hidden_states)
        ff_out = ff.DenseReluDense(ff_in)
        ff_out = ff.dropout(ff_out)

        residual = _apply_shortcut(residual, getattr(self, "mlp_shortcut_Q", None))
        residual = _match_last_dim(residual, ff_out.shape[-1])
        hidden_states = residual + ff_out
        hidden_states = _maybe_clamp_fp16(hidden_states)

        # ---------------------------
        # Pack outputs EXACTLY like HF T5Block
        # ---------------------------
        out = (hidden_states,)

        if self.is_decoder:
            # present_key_value must be ONE object: (self_kv, cross_kv)
            if use_cache:
                present_key_value = _flatten_present_kv(self_present_kv, cross_present_kv)
                out = out + (present_key_value,)

            # then self-attn position bias
            out = out + (position_bias,)

            if output_attentions:
                out = out + (self_attn_w,)

            # then cross-attn position bias (if cross-attn ran)
            if do_cross_attention:
                out = out + (encoder_decoder_position_bias,)
                if output_attentions:
                    out = out + (cross_attn_w,)
        else:
            # encoder block: no cross-attn
            if use_cache:
                # encoder cache is just self kv
                out = out + (self_present_kv,)

            out = out + (position_bias,)
            if output_attentions:
                out = out + (self_attn_w,)

        return out




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
    
    # ----------------------------
    # Cross-attention (encoder: none)
    # These exist only to satisfy LayerAdapter abstract interface.
    # ----------------------------
    def has_cross_attention(self) -> bool:
        return False

    def get_cross_attention_layernorm(self) -> Module:
        raise RuntimeError("Encoder T5Block has no cross-attention.")

    def get_cross_attention_q_input(self) -> Linear:
        raise RuntimeError("Encoder T5Block has no cross-attention.")

    def get_cross_attention_kv_inputs(self) -> Sequence[Linear]:
        raise RuntimeError("Encoder T5Block has no cross-attention.")

    def get_cross_attention_output(self) -> Linear:
        raise RuntimeError("Encoder T5Block has no cross-attention.")



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
    
    def has_cross_attention(self) -> bool:
        return True



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

    # -------------------------
    # Internal helpers
    # -------------------------
    def _enc(self):
        return self._model.get_encoder() if hasattr(self._model, "get_encoder") else self._model.encoder

    def _dec(self):
        return self._model.get_decoder() if hasattr(self._model, "get_decoder") else self._model.decoder

    # -------------------------
    # Layer accessors
    # -------------------------
    def get_encoder_layers(self) -> list[LayerAdapter]:
        enc = self._enc()
        return [T5BlockLayerAdapter(layer) for layer in enc.block]

    def get_decoder_layers(self) -> list[LayerAdapter]:
        dec = self._dec()
        return [T5DecoderBlockLayerAdapter(layer) for layer in dec.block]

    # legacy SliceGPT API → encoder stack
    def get_layers(self) -> list[LayerAdapter]:
        return self.get_encoder_layers()

    # -------------------------
    # Raw layer access (used by catchers)
    # -------------------------
    def get_raw_layer_at(self, index: int):
        return self._enc().block[index]

    def set_raw_layer_at(self, index: int, new_layer):
        self._enc().block[index] = new_layer

    def get_raw_encoder_layer_at(self, index: int):
        return self._enc().block[index]

    def set_raw_encoder_layer_at(self, index: int, new_layer):
        self._enc().block[index] = new_layer

    def get_raw_decoder_layer_at(self, index: int):
        return self._dec().block[index]

    def set_raw_decoder_layer_at(self, index: int, new_layer):
        self._dec().block[index] = new_layer

    # ---- compression layer replacement ----

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        if not isinstance(layer, T5Block):
            raise TypeError(f"Expected T5Block, got {type(layer)}")

        # Detect relative position bias in the self-attention sublayer
        has_rpb = bool(getattr(layer.layer[0].SelfAttention, "has_relative_attention_bias", False))

        # IMPORTANT: T5Block structure depends on config.is_decoder.
        # If we build with is_decoder=False, we get encoder-style (no cross-attn).
        # If we build with is_decoder=True, we get decoder-style (with cross-attn).
        cfg = copy.deepcopy(self.config)
        cfg.is_decoder = bool(layer.is_decoder)
        # (Usually already true for T5; harmless if set)
        cfg.is_encoder_decoder = True

        ctor = self.compressed_layer_type
        sig = inspect.signature(ctor.__init__)
        params = sig.parameters

        kwargs = {"has_relative_attention_bias": has_rpb}
        if "layer_idx" in params:
            kwargs["layer_idx"] = layer_idx

        compressed = ctor(cfg, **kwargs)

        # Now state_dict layouts match (encoder vs decoder)
        compressed.load_state_dict(layer.state_dict(), strict=True)

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
