from types import MethodType
from typing import Tuple, Optional, Union, List

import math
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.utils import logging
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

logger = logging.get_logger(__name__)


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

# The following sketched forward pass is built upon the Huggingface Transformer library's PyTorch implementation of qwen2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py.
# NOTE: Qwen2 has been updated beyond the transformer requirements of this project. As such, the following code was taken directly from the huggingface package within this python environment.

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # NOTE: THE FOLLOWING LINE IS THE ONLY CHANGE
        attn_output = sketch(attn_weights, value_states)
        if isinstance(attn_output, tuple):
            attn_output, sketching_info_dict = attn_output
            if hasattr(self, 'sketching_info_dicts'):
                self.sketching_info_dicts.append(sketching_info_dict)
            else:
                self.sketching_info_dicts = [sketching_info_dict]

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward
