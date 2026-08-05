import torch
from torch import nn
class GPTEmbedding(nn.Module):
    """
        GPT embedding layer consisting of:
            - Token embeddings
            - Positional embeddings
            - Dropout
        """
    def __init__(self, config):
        super().__init__()

        # Token Embedding - a vector representation of the token.
        # Initialized with random small numbers
        # During the training process the embeddings corresponding to
        # the words(tokens) with close meanings will get closer (<distance)
        # Embedding matrix has size (vocab_size X emb_dim). When input x is proceeded during forward pass,
        # the lookup happens. Ex., input token=24, then lookup gives 24-th row of Embedding matrix, vector of size=emb_dim
        # In other words for each token in the vocabulary there is a corresponding emb_dim-size vector

        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)

        # pos_emb - encode the position of the token in the text.
        # In GPT model absolute position is used, but can be also relative
        # in other models. It's a randomly initialized matrix of size length of the embedding=emb_dim
        # and for each token in the text sample=context_length - batch row. The position is spread
        # along the batch, i.e. the same pos is added to each row in the batch
        self.position_embedding = nn.Embedding(config.context_length, config.emb_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
                input_ids: (batch_size, seq_length)
                        Returns:
                                embeddings: (batch_size, seq_length, emb_dim)
                """

        batch_size, content_length = x.shape

        # Creating the indexes for pos emb, from 0 to context_length-1
        # shape (content_length, )
        positions = torch.arange(
            content_length,
            device=x.device
        )

        # Shape(batch_size, content_length, emb_dim)
        token_embeddings = self.token_embedding(x)

        # Shape (content_length, emb_dim)
        position_embeddings = self.position_embedding(positions)

        # Broadcasting:
        # (batch_size, seq_length, emb_dim)
        embeddings = token_embeddings + position_embeddings

        return self.dropout(embeddings)


