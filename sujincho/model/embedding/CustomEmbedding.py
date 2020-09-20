import torch
from torch import nn
from .token import TokenEmbedding
from .positional import PositionalEmbedding
from .weekday import WeekdayEmbedding
from .hour import HourEmbedding

class CustomEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information
    3. WeekdayEmbedding : adding weekday information
    4. HourEmbedding : adding time information
    sum of all these features are output of Embedding
    """

    def __init__(self, d_embedding, d_model, device=None, pad_idx=0):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.device = device

        self.token = nn.Linear(1, d_embedding).to(device)
        self.position = PositionalEmbedding(d_embedding=d_embedding, d_model=d_model, device=self.device)
        self.weekday = WeekdayEmbedding(d_embedding=d_embedding, d_model=d_model, device=self.device)
        self.hour = HourEmbedding(d_embedding=d_embedding, d_model=d_model, device=self.device)

        self.linear_layer = nn.Linear(d_embedding, d_model)#.to(device)
        self.norm = nn.LayerNorm(d_model)
        # To concat
        # self.concat_linear_layer = nn.Linear(d_embedding * 3, d_embedding)

    def forward(self, sequence, weekday, hour):
        x = self.linear_layer(self.token(sequence.unsqueeze(2))) + self.position(sequence) + self.hour(hour) + self.weekday(weekday)
        # token_emb = self.linear_layer(self.token(sequence.unsqueeze(2)))
        # position_emb = self.position(sequence)
        # hour_emb = self.hour(hour)
        # weekday_emb = self.weekday(weekday)
        # emb_cat = torch.cat((token_emb, position_emb, hour_emb, weekday_emb), dim=2)
        # emb_cat = emb_cat.view(8, -1, d_embedding * 3)
        # x = concat_linear(emb_cat).view(8, d_embedding * 3, -1)
        return self.norm(x)