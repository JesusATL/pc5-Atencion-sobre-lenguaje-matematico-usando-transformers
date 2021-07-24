import torch
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
from typing import Tuple

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    """
    La entrada consiste de vectores consultas Q, K, V y vectores claves de dimensión d_k, y valores de dimensión d_v. Luego se sigue una serie de operaciones, resultando la salida como una suma ponderada de los valores.
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.0):
        super().__init__()
        self.temperature = temperature
        # el dropout es la probabilidad de eliminar las conexiones en la FF para evitar overfitting
        self.dropout = nn.Dropout(p=attn_dropout) if attn_dropout != 0 else None
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tuple[Tensor]:
        # bmm realiza un producto MatMul(Q, K)
        attn: Tensor = torch.bmm(input=q, mat2=k.transpose(dim0=1, dim1=2))
        # Scale
        attn: Tensor = attn / self.temperature

        # Mask
        if mask is not None:
            attn = torch.masked_fill(mask, -np.inf)
        # SoftMax
        attn: Tensor = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # MatMul
        output = torch.bmm(attn, v)
        return output, attn
