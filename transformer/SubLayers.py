""" Define the sublayers in encoder/decoder layer """
import numpy as np
from torch._C import TensorType
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention
from typing import Tuple

__author__ = "Yu-Hsiang Huang"


class MultiHeadAttencion(nn.Module):
    """ Multi-Head Attention module: Se trata se mejorar la función de atención al realizar procesamiento paralelo."""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # normalizacion
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.sqrt(d_k))  # QK^T/sqrt(d_k)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal(self.fc.weight)

        self.dropout = nn.Dropout(dropout) if dropout != 0 else None

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tuple[Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual: Tensor = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output: Tensor = output.view(n_head, sz_b, len_q, d_v)
        output = (output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1))  # b x lq x (n*dv)
        output = self.fc(output)

        if self.dropout is not None:
            output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.0):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout) if dropout != 0 else None

    def forward(self, x: Tensor) -> Tensor:
        residual: Tensor = x
        output: Tensor = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
