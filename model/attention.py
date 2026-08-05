import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout=0, qkv_bias=False):
    super().__init__()
    self.d_out = d_out

    # Instead of initializing of W matrices with random values
    # we use nn.Linear, which are equivalent to a matrix multiplication
    # if we disable the bias units
    # Another big advantage of using nn.Linear over our manual
    # nn.Parameter(torch.rand(...) approach is that nn.Linear has a
    # preferred weight initialization scheme, which leads to more
    # stable model training
    self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    self.dropout = nn.Dropout(p=dropout)

    # register model's buffer, which isn't a model parameter
    # but be a part of the model and would be on the same device(CPU,
    # GPU) as the model itself avoiding device mismatch errors
    self.register_buffer(
        "mask",
        torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

  def forward(self, x):
    # b - batch size
    # num_token - number of tokens
    # d_in - the length of the embedding vector
    b, num_tokens, d_in = x.shape
    keys = self.W_key(x)
    queries = self.W_query(x)
    values = self.W_value(x)

    attn_scores = queries @ keys.transpose(1,2) # the dims 1 and 2 are transposed, but batch position(=0) is kept on place

    # hiding future values with triangular mask, above diagonal values=-inf
    # .masked_fill_ - underscore funcs in PyTorch are performed in
    # place, w/t memory copies
    attn_scores.masked_fill_((self.mask==1)[:num_tokens,:num_tokens], -torch.inf)

    attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
    attn_weights = self.dropout(attn_weights)
    context_vecs = attn_weights @ values

    return context_vecs

  """The division by keys.shape[-1]**0.5 is a technique called 
  'scaling' in the 'Scaled Dot-Product Attention'. Its purpose is 
  to counteract the effect of large dot products when the dimension 
  of the key vectors (d_k, which is keys.shape[-1]) is high. Without 
  this scaling, the dot products can become very large in magnitude, 
  pushing the softmax function into regions where its gradients are 
  extremely small, leading to vanishing gradients and hindering 
  effective learning during model training."""
