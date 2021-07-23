from typing import Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"


def get_non_pad_mask(seq: Tensor):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position: int,
                                d_hid: int,
                                padding_idx: Tensor = None) -> Tensor:
    """ Sinusoid position encoding table """
    def cal_angle(position: int, hid_idx: int) -> float:
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position: int) -> list:
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i + 1

    if padding_idx is not None:
        # zero vector for padding dimesion
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k: Tensor, seq_q: Tensor) -> Tensor:
    """ For masking out the padding part of key sequence. """
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq: Tensor) -> Tensor:
    """ For masking out the subsequent info. """
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8),
        diagonal=1
    )
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """
    A encoder model with self attention mechanism.
    ---------------------
    [Add & Norm
    Feed-Forward]
        ↑
    [Add & Norm
    Multi-Head Attention]
    ---------------------
        ↑
    [Positional Encoding]
        ↑
    [Input Embedding]
    """

    def __init__(
            self,
            n_src_vocab: int,
            len_max_seq: int,
            d_word_vec: int,
            n_layers: int,
            n_heads: int,
            d_k: int,
            d_v: int,
            d_model: int,
            d_inner: int,
            dropout: float = 0.0):
        super().__init__()
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        # Debido a que el codigo se ejecuta en paralelo, hay que indicarle la posicion relativa
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(
                n_position, d_word_vec, padding_idx=Constants.PAD
            ),
            freeze=True
        )
        # Nx, en el paper hay 6 encoder y decoders
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, d_k, d_v, dropout=dropout) for _ in range(n_layers)
            ]
        )

    def forward(self,
                src_seq: Tensor,
                src_pos: Tensor,
                return_attns: bool = False) -> Tuple[Tensor]:
                
        enc_slf_attn_list: Tuple[Tensor] = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward, aqui esta las suma de la codificacion posicional y el embedding de entrada
        enc_output: Tensor = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask,
                slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list

        return (enc_output, )

class Decoder(nn.Module):
    pass