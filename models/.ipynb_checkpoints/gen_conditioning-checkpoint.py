"""
Genre condition for the model 
inspired by AudioLDM text conditioning pipeline, but genres instead

AudioLDM code:
    https://github.com/haoheliu/AudioLDM

in AudioLDM, they use text to condition, i.e text to CLAP/T5 encoder to token embeddings to cross-attention UNet. here we use genre to condition, i.e genre ID to embedding to token expansion to cross-attention UNet

we also added classifier-free guidance with a NULL genre token
"""

import torch
import torch.nn as nn


class GenreCondition(nn.Module):
    def __init__(self):
        super().__init__()
        #note: change this as the number of genres will change if we choose to use fma_medium
        self.num_genres= 8
        self.null_genre_id = 8
        self.embedding_length = self.num_genres + 1
        #if conditioning is poor maybe we scale this up, if it slow but good maybe drop to 4
        self.tokens = 8
        #dimension used in AudioLDM but subject to change based on performance
        self.embedding_dimension = 512
        #the embedding of the genre
        self.genre_embedding = nn.Embedding(self.embedding_length, self.embedding_dimension)
        # putting the genre embedding into token sequence
        self.genre_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension * self.tokens)

    def forward(self, genre_ids: torch.Tensor) -> torch.Tensor:
        #genre to embedding
        embedding = self.genre_embedding(genre_ids)
        #embedding to token
        final_tokens = self.genre_projection(embedding)
        # output should be (batch count, 8, 512)
        batch_count = genre_ids.shape[0]
        final_tokens = final_tokens.reshape(batch_count, self.tokens, self.embedding_dimension)
        return final_tokens