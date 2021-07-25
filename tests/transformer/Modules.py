import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout != 0 else None
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        print ("*"*30)
        print (type(q))
        print (type(k))
        print (q.size())
        print (k.size())
        print ("*"*30)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
