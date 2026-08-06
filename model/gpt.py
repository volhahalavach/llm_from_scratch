import torch.nn as nn

import transformer
import normalization
import embeddings

class GPTModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.embeddings = embeddings.GPTEmbedding(config)

    self.transformer_blocks = nn.Sequential(
        *[transformer.TransformerBlock(config) for _ in range(config.n_heads)]    )

    self.final_norm = normalization.LayerNorm(config.emb_dim)

    self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)


  def forward(self, x):
    x = self.embeddings(x)
    x = self.transformer_blocks(x)
    x = self.final_norm(x)
    output = self.out_head(x)
    return output