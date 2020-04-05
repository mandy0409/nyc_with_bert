import torch
from torch import nn

class PositionalEmbedding(nn.Module):

    def __init__(self, d_embedding, d_model, max_len=12, device=None):
        super().__init__()
        self.max_len = max_len
        self.device = device

        self.embedding = nn.Embedding(self.max_len, d_embedding)
        self.fep_linear = nn.Linear(d_embedding, d_model) # For factorized embedding parameterization (from ALBERT)

    def forward(self, x):
        position = torch.arange(self.max_len).to(self.device)
        position = self.fep_linear(self.embedding(position))
        
        return position.repeat(x.size(0), 1, 1)