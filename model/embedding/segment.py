import torch
from torch import nn

class HolidayEmbedding(nn.Module):

    def __init__(self, d_embedding, d_model, max_len=300, device=None):
        super().__init__()

        self.device = device

        self.embedding = nn.Embedding(2, d_embedding) # Holiday embedding
        self.fep_linear = nn.Linear(d_embedding, d_model) # For factorized embedding parameterization (from ALBERT)

    def forward(self, x):

        holiday = self.fep_linear(self.embedding(x))

        return segment.repeat(x.size(0), 1, 1)