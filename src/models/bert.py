from types import MethodType
from typing import Tuple, Optional, List

import math
import torch
from torch.nn.functional import softmax


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
        model.transformer.layer[i].attention.forward = MethodType(sketched_forward(sketch), model.transformer.layer[i].attention)


# ================
# Sketched Forward
# ================

# This sketched function is built upon the Huggingface Transformer library's PyTorch implementation of distilbert: https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py 

def sketched_forward(sketch):
    """
    Returns the distilgpt2 forward function with the sketch function used to sketch the matrix multiplication AV.

    Args:
        sketch (function):
            A sketching function that takes two matrices as inputs and returns their sketched matrix multiplication.
    """

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
        ) -> Tuple[torch.Tensor, ...]:
            """
            Parameters:
                query: torch.tensor(bs, seq_length, dim)
                key: torch.tensor(bs, seq_length, dim)
                value: torch.tensor(bs, seq_length, dim)
                mask: torch.tensor(bs, seq_length)

            Returns:
                weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
                seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
            """
            bs, q_length, dim = query.size()
            k_length = key.size(1)
            # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
            # assert key.size() == value.size()

            dim_per_head = self.dim // self.n_heads

            mask_reshp = (bs, 1, 1, k_length)

            def shape(x: torch.Tensor) -> torch.Tensor:
                """separate heads"""
                return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

            def unshape(x: torch.Tensor) -> torch.Tensor:
                """group heads"""
                return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

            q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
            k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
            v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

            q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
            scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
            mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
            scores = scores.masked_fill(
                mask, torch.tensor(torch.finfo(scores.dtype).min)
            )  # (bs, n_heads, q_length, k_length)

            weights = softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
            weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

            # Mask heads if we want to
            if head_mask is not None:
                weights = weights * head_mask

            # NOTE: THE FOLLOWING LINE IS THE ONLY CHANGE
            context = sketch(weights, v)  # (bs, n_heads, q_length, dim_per_head)
            if isinstance(attn_output, tuple):
                attn_output, sketching_info_dict = attn_output
                if hasattr(self, 'sketching_info_dicts'):
                    self.sketching_info_dicts.append(sketching_info_dict)
                else:
                    self.sketching_info_dicts = [sketching_info_dict]

            context = unshape(context)  # (bs, q_length, dim)
            context = self.out_lin(context)  # (bs, q_length, dim)

            if output_attentions:
                return (context, weights)
            else:
                return (context,)
    
    return forward
