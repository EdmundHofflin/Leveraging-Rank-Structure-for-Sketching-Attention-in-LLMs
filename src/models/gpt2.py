from loguru import logger

from types import MethodType
from typing import Tuple, Optional, Union, List

import torch

from utils.transformer_utils import sketch_scaled_dot_product_attention


# ========
# Patching
# ========

def patch_attention(model, sketch, patch_idxs : List[int]):
    """
    Patches the attention layers of a given transformer model by augmenting their forward pass. The patched layers use matrix sketching to approximate AV and thereby improve computational efficiency.
    
    Args:
        model:
            Callable model to evaluate. Internal attention layers must be accessible.
        sketch (function):
            A sketching function that takes two matrices as inputs and returns their sketched matrix multiplication.
        patch_idxs (List[int]):
            A list of ints corresponding to the layers of the model to patch.
    """

    for i in patch_idxs:
        model.transformer.h[i].attn.forward = MethodType(sketched_forward(sketch), model.transformer.h[i].attn)


# ================
# Sketched Forward
# ================

# The following sketched forward pass is built upon the Huggingface Transformer library's PyTorch implementation of gpt2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

def sketched_forward(sketch):
    """
    Returns the gpt2 attention module with the sketch function used to sketch the matrix multiplication AV within the forward pass.

    Args:
        sketch (function):
            A sketching function that takes two matrices as inputs and returns their sketched matrix multiplication.

    """

    sdpa = sketch_scaled_dot_product_attention(sketch)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`GPT2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        bsz, q_len, _ = hidden_states.size()

        # Initial attention projections
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Optional kv caching
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        # FIXME: THE FOLLOWING PART IS COMMENTED OUT AS self.require_contiguous_qkv IS NOT AN ATTRIBUTE OF GPT2ATTENTION?SS
        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        # if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
        #     query = query.contiguous()
        #     key = key.contiguous()
        #     value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

        # NOTE: THE FOLLOWING LINE IS THE ONLY CHANGE
        attn_output = sdpa(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        # Tuple signifies that sketching_info has flagged, so we must save that data
        if isinstance(attn_output, tuple):
            attn_output, sketching_info_dict = attn_output
            if hasattr(self, 'sketching_info_dicts'):
                self.sketching_info_dicts.append(sketching_info_dict)
            else:
                self.sketching_info_dicts = [sketching_info_dict]

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present, None

    return forward