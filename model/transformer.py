import normalization
import attention
import feedforward

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = attention.MultiHeadAttention(
        d_in=config.emb_dim,
        d_out=config.emb_dim,
        context_length=config.context_length,
        num_heads=config.num_heads,
        dropout=config.dropout,
        qkv_bias=config.qkv_bias)
    self.feed_fwd=feedforward.FeedForward(config)
    self.norm_attn = normalization.LayerNorm(config.emb_dim)
    self.norm_ffd = normalization.LayerNorm(config.emb_dim)
    self.drop_shortcut = nn.Dropout(config.dropout)

  def forward(self, x):
    # saving shortcut
    shortcut = x

    # Attention block
    x = self.norm_attn(x)
    x = self.attn(x)
    x = self.drop_shortcut(x)
    x = x + shortcut    # add the shortcut = original input

    # saving shortcut again
    shortcut = x

    # Feed Forward block
    x = self.norm_ffd(x)
    x = self.feed_fwd(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    return x
