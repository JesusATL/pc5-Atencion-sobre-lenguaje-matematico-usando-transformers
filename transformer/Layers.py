from torch.functional import Tensor
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttencion, PositionwiseFeedForward
from typing import Tuple

__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    """
    Compose with two layers:
    ---------------------
    [Add & Norm
    Feed-Forward]
        ↑
    [Add & Norm
    Multi-Head Attention]
    ---------------------
    """

    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttencion(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self,
                enc_input: Tensor,
                non_pad_mask: Tensor = None,
                slf_attn_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        # la entrada codificada se triplica en los tras matriz Q, K, V
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output: Tensor = enc_output * non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output * non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """
    Compose with three layers
    ---------------------
    [Add & Norm
    Feed-Forward]
        ↑
    [Add & Norm
    Multi-Head Attention]
        ↑
    [Add & Norm
    Masked Multi-Head Attention]
    ---------------------
        ↑
    [Positional Encoding]
        ↑
    [Output Embedding]
    """

    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.0):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttencion(n_head, d_model, d_k, d_v, dropout)
        self.enc_attn = MultiHeadAttencion(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)

    def forward(self,
                dec_input: Tensor,
                enc_output: Tensor,
                non_pad_mask: Tensor = None,
                slf_attn_mask: Tensor = None,
                dec_enc_attn_mask: Tensor = None) -> Tuple[Tensor]:
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        dec_output: Tensor = dec_output * non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )

        dec_output: Tensor = dec_output * non_pad_mask
        dec_output = self.pos_ffn(dec_output)
        dec_output = dec_output * non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
