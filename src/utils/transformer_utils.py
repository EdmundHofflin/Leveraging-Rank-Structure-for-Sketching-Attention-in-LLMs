import math
import torch


# =====================================
# Sketched Scaled Dot Product Attention
# =====================================

# The following implementation is built upon the Torch library's equivalent implementation of scaled dot product attention: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

def sketch_scaled_dot_product_attention(sketch):
    """
    Returns the scaled dot product attention function with the sketch function used to sketch the matrix multiplication AV.
    
    Args:
        sketch (function):
            A sketching function that takes two matrices as inputs and returns their sketched matrix multiplication.
        info (bool):
            Whether to return all information from the sketching.
    """

    def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return sketch(attn_weight, value)

    return scaled_dot_product_attention
