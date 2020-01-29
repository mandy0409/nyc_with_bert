# coding: utf-8
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .embedding.CustomEmbedding import CustomEmbedding


class littleBert(nn.Module):
    def __init__(self, vocab_num, pad_idx=0, bos_idx=1, eos_idx=2, max_len=300, d_model=512, d_embedding=256, n_head=8, 
                 dim_feedforward=2048, n_layers=10, dropout=0.1, device=None):
        super(littleBert, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.dropout = nn.Dropout(dropout)
        self.device = device

        # Source embedding part
        self.src_embedding = CustomEmbedding(vocab_num, d_embedding, d_model, device=self.device, pad_idx=self.pad_idx, max_len=self.max_len)

        # Transformer
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward,
                activation='gelu', dropout=dropout) for i in range(n_layers)])

        # Output of model
        self.linear = nn.Linear(d_model, 30)

    def forward(self, title_sequence, body_sequence, ans_sequence):
        sep_token = torch.tensor([4]).repeat(title_sequence.size(0), 1).to(self.device) # Replace 0 to other number
        cls_token = torch.tensor([5]).repeat(title_sequence.size(0), 1).to(self.device) # Replace 0 to other number
        sequence = torch.cat((cls_token, title_sequence, sep_token, body_sequence, sep_token, ans_sequence, sep_token), dim=1)
        encoder_out = self.src_embedding(sequence).transpose(0, 1)
        src_key_padding_mask = sequence == self.pad_idx

        for i in range(len(self.encoders)):
            encoder_out = self.encoders[i](encoder_out, src_key_padding_mask=src_key_padding_mask)

        outputs = self.linear(encoder_out)

        return outputs

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1, 
            activation="relu"):
        
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
