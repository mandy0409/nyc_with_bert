import torch
from torch import nn
from .token import TokenEmbedding
from .positional import PositionalEmbedding
from .weekday import WeekdayEmbedding
from .hour import HourEmbedding
from .location import LocationEmbedding

class CustomEmbedding(nn.Module):
    def __init__(self, d_embedding, d_model, device=None, pad_idx=0):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.device = device

        self.token = nn.Linear(1, d_embedding).to(device)
        self.position = PositionalEmbedding_kh(d_embedding=d_embedding, d_model=d_model, device=self.device)
        self.weekday = WeekdayEmbedding(d_embedding=d_embedding, d_model=d_model, device=self.device)
        self.hour = HourEmbedding(d_embedding=d_embedding, d_model=d_model, device=self.device)
        self.location = LocationEmbedding(d_embedding=d_embedding, d_model=d_model, device=self.device)

        self.linear_layer = nn.Linear(d_embedding, d_model)#.to(device)
        self.norm = nn.LayerNorm(d_model)
        
        # Add: To concat
        self.concat_linear_layer = nn.Linear(d_embedding * 3, d_embedding)

    def forward(self, sequence, weekday, hour):
        # x = self.linear_layer(self.token(sequence.unsqueeze(2))) 
        # + self.position(sequence) 
        # + self.hour(hour) 
        # + self.weekday(weekday)
        # + self.location(location)
        
        # Add: To concat
        token_emb = self.linear_layer(self.token(sequence.unsqueeze(2)))
        position_emb = self.position(sequence)
        hour_emb = self.hour(hour)
        weekday_emb = self.weekday(weekday)
        location_emb = self.location(location)

        emb_cat = torch.cat((token_emb, position_emb, hour_emb, weekday_emb, location_emb), dim=2)
        emb_cat = emb_cat.view(8, -1, d_embedding * 3)
        x = concat_linear(emb_cat).view(8, d_embedding * 3, -1)

        return self.norm(x)