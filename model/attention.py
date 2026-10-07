import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0, qkv_bias=False):
      super().__init__()
      assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

      self.d_out = d_out
      self.num_heads = num_heads
      self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim
      self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
      self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
      self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
      self.out_proj = nn.Linear(d_out, d_out)  # linear layer to combine heads output
      self.dropout = nn.Dropout(p=dropout)
      self.register_buffer("mask",
                           torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
      # b - batch size
      # num_token - number of tokens
      # d_in - the length of the embedding vector
      b, num_tokens, d_in = x.shape
      keys = self.W_key(x)
      queries = self.W_query(x)
      values = self.W_value(x)

      # Tensor.view() - Reshaping keys, queries, values from 3dim to 4dim
      # d_out is a big num and d_out=num_head*head_dim
      keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
      queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
      values = values.view(b, num_tokens, self.num_heads, self.head_dim)

      # transpose from shape (b,num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
      # so the last 2 dimensions could be multiplied
      keys = keys.transpose(1, 2)
      queries = queries.transpose(1, 2)
      values = values.transpose(1, 2)

      # compute attn scores for each head
      attn_scores = queries @ keys.transpose(2, 3)

      # hiding the future words, ie values over main diagonal
      attn_scores.masked_fill_(((self.mask == 1)[:num_tokens, :num_tokens]), -torch.inf)
      # normalization
      attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
      # dropout
      attn_weights = self.dropout(attn_weights)

      # calculating context vector
      context_vec = (attn_weights @ values).transpose(1,
                                                      2)  # transpose to initial shape (b,num_tokens, num_head, head_dim)

      # Tensor.contiguous() - Returns a contiguous in memory tensor containing the same data as self tensor
      context_vec = context_vec.contiguous().view(b, num_tokens,
                                                  self.d_out)  # we need to return one long context vector for all heads (concatenate context vectors from all heads)

      # Add an optional linear projection
      context_vec = self.out_proj(context_vec)

      return context_vec


"""The self.out_proj in the MultiHeadAttention class applies an additional
 linear projection to the concatenated context vectors from all the 
 attention heads. This step is crucial for several reasons:

* Dimensionality Matching: Each attention head produces an output of 
head_dim. When you have num_heads heads, their outputs are typically 
concatenated, resulting in a vector of size num_heads * head_dim, which 
is d_out. The self.out_proj then transforms this concatenated vector 
back into the d_out dimension, allowing the output of the multi-head 
attention block to have the same dimensionality as its input, which is 
essential for residual connections and stacking multiple layers.
* Information Combination: It provides a learnable way to linearly 
combine the diverse information captured by different attention heads. 
Each head might focus on different aspects of the input sequence, and 
this projection allows the model to weigh and integrate these different 
perspectives.
* Increased Model Capacity: Adding this linear layer increases the 
model's capacity to learn complex relationships and transformations on 
the aggregated information from the attention heads."""

"""The division by keys.shape[-1]**0.5 is a technique called 
  'scaling' in the 'Scaled Dot-Product Attention'. Its purpose is 
  to counteract the effect of large dot products when the dimension 
  of the key vectors (d_k, which is keys.shape[-1]) is high. Without 
  this scaling, the dot products can become very large in magnitude, 
  pushing the softmax function into regions where its gradients are 
  extremely small, leading to vanishing gradients and hindering 
  effective learning during model training."""