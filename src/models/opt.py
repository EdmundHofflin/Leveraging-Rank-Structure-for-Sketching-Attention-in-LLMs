from loguru import logger

from types import MethodType
from typing import Tuple, Optional, List

import torch
from torch import nn

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

    logger.trace(f"Starting patch_attention: {model=}, {sketch=}, {patch_idxs=}")

    for i in patch_idxs:
        model.model.decoder.layers[i].self_attn.forward = MethodType(sketched_forward(sketch), model.model.decoder.layers[i].self_attn)

    logger.trace(f"Successful patch_attention.")


# ================
# Sketched Forward
# ================

# The following sketched forward pass is built upon the Huggingface Transformer library's PyTorch implementation of opt: https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py

def sketched_forward(sketch):
    """
    Returns the gpt2 attention module with the sketch function used to sketch the matrix multiplication AV within the forward pass.

    Args:
        sketch (function):
            A sketching function that takes two matrices as inputs and returns their sketched matrix multiplication.
    """

    logger.trace(f"Starting sketched_forward: {sketch=}")

    sdpa = sketch_scaled_dot_product_attention(sketch)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions or layer_head_mask is not None:
            logger.warning_once(
                "OPTModel is using SDPA attention, which currently does not support output_attentions=True."
                'failing back to eager attention. remove warning using attn_implementation="eager".'
            )

            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                key_value_states=key_value_states,
            )  # TODO after merge add position_ids=position_ids
        is_cross_attention = key_value_states is not None

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, -1, bsz)

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # shape now is (bsz, num_heads, seq_len, head_dim), all are continuous

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        # NOTE: THE FOLLOWING LINE IS THE ONLY CHANGE
        attn_output = sdpa(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            # this model uses the scaling factor in the query projection for some reason, but not in Q@K^T
            # so we need to scale to remove scaling in SDPA to have similar results with eager.
            # Maybe needs a change in the model to remove scaling in query projection
            scale=1.0,
        )
        if isinstance(attn_output, tuple):
            attn_output, sketching_info_dict = attn_output
            if hasattr(self, 'sketching_info_dicts'):
                self.sketching_info_dicts.append(sketching_info_dict)
            else:
                self.sketching_info_dicts = [sketching_info_dict]

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value
    
    logger.trace(f"Successful sketched_forward: {sketch=}")

    return forward
