from types import MethodType
from typing import Tuple, Optional, Union, List

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

    for i in patch_idxs:
        model.model.layers[i].self_attn.forward = MethodType(sketched_forward(sketch), model.model.layers[i].self_attn)


# ================
# Sketched Forward
# ================

# The following sketched forward pass is built upon the Huggingface Transformer library's PyTorch implementation of xglm: https://github.com/huggingface/transformers/blob/main/src/transformers/models/xglm/modeling_xglm.py

def sketched_forward(sketch):
    """
    Returns the gpt2 attention module with the sketch function used to sketch the matrix multiplication AV within the forward pass.

    Args:
        sketch (function):
            A sketching function that takes two matrices as inputs and returns their sketched matrix multiplication.
    """

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            layer_head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            """Input shape: Batch x Time x Channel"""

            # if key_value_states are provided this layer is used as a cross-attention layer
            # for the decoder
            is_cross_attention = key_value_states is not None

            bsz, tgt_len, _ = hidden_states.size()

            # get query proj
            query_states = self.q_proj(hidden_states) * self.scaling
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

            proj_shape = (bsz * self.num_heads, -1, self.head_dim)
            query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)

            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
            if attn_weights.dtype == torch.float16:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
            else:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            if layer_head_mask is not None:
                if layer_head_mask.size() != (self.num_heads,):
                    raise ValueError(
                        f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                        f" {layer_head_mask.size()}"
                    )
                attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if output_attentions:
                # this operation is a bit awkward, but it's required to
                # make sure that attn_weights keeps its gradient.
                # In order to do so, attn_weights have to be reshaped
                # twice and have to be reused in the following
                attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
            else:
                attn_weights_reshaped = None

            attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

            # NOTE: THE FOLLOWING LINE IS THE ONLY CHANGE
            attn_output = sketch(attn_probs, value_states)
            if isinstance(attn_output, tuple):
                attn_output, sketching_info_dict = attn_output
                if hasattr(self, 'sketching_info_dicts'):
                    self.sketching_info_dicts.append(sketching_info_dict)
                else:
                    self.sketching_info_dicts = [sketching_info_dict]

            if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            attn_output = attn_output.transpose(1, 2)

            # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
            # partitioned aross GPUs when using tensor-parallelism.
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

            attn_output = self.out_proj(attn_output)

            return attn_output, attn_weights_reshaped, past_key_value

    return forward
