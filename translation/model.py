import math
from typing import Callable

import torch.nn as nn
from torch import Tensor

from translation.layers import (
    DictionaryEncoding,
    Embedding,
    FeedForward,
    MultiHeadAttention,
    PositionalEncoding,
    ScaleNorm,
    clone,
)

Sublayer = Callable[[Tensor], Tensor]


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = ScaleNorm(embed_dim**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: Sublayer) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 2)

    def forward(
        self, src_encs: Tensor, src_mask: Tensor | None = None, dict_mask: Tensor | None = None
    ) -> Tensor:
        src_encs = self.sublayers[0](
            src_encs, lambda x: self.self_attn(x, x, x, src_mask, dict_mask)
        )
        return self.sublayers[1](src_encs, self.ff)


class Encoder(nn.Module):
    def __init__(
        self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float, num_layers: int
    ):
        super(Encoder, self).__init__()
        self.layers = clone(EncoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        self.norm = ScaleNorm(embed_dim**0.5)

    def forward(
        self, src_embs: Tensor, src_mask: Tensor | None = None, dict_mask: Tensor | None = None
    ) -> Tensor:
        src_encs = src_embs
        for layer in self.layers:
            src_encs = layer(src_encs, src_mask, dict_mask)
        return self.norm(src_encs)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.crss_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.sublayers = clone(SublayerConnection(embed_dim, dropout), 3)

    def forward(
        self,
        src_encs: Tensor,
        tgt_encs: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        m = src_encs
        tgt_encs = self.sublayers[0](tgt_encs, lambda x: self.self_attn(x, x, x, tgt_mask))
        tgt_encs = self.sublayers[1](tgt_encs, lambda x: self.crss_attn(x, m, m, src_mask))
        return self.sublayers[2](tgt_encs, self.ff)


class Decoder(nn.Module):
    def __init__(
        self, embed_dim: int, ff_dim: int, num_heads: int, dropout: float, num_layers: int
    ):
        super(Decoder, self).__init__()
        self.layers = clone(DecoderLayer(embed_dim, ff_dim, num_heads, dropout), num_layers)
        self.norm = ScaleNorm(embed_dim**0.5)

    def forward(
        self,
        src_encs: Tensor,
        tgt_embs: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        tgt_encs = tgt_embs
        for layer in self.layers:
            tgt_encs = layer(src_encs, tgt_encs, src_mask, tgt_mask)
        return self.norm(tgt_encs)


class Model(nn.Module):
    def __init__(
        self,
        vocab_dim: int,
        embed_dim: int,
        ff_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
    ):
        super(Model, self).__init__()
        self.encoder = Encoder(embed_dim, ff_dim, num_heads, dropout, num_layers)
        self.decoder = Decoder(embed_dim, ff_dim, num_heads, dropout, num_layers)
        self.out_embed = Embedding(embed_dim, math.ceil(vocab_dim / 8) * 8)
        self.src_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))
        self.tgt_embed = nn.Sequential(self.out_embed, PositionalEncoding(embed_dim, dropout))
        self.dpe_embed = nn.Sequential(self.out_embed, DictionaryEncoding(embed_dim))

    def encode(
        self,
        src_nums: Tensor,
        src_mask: Tensor | None = None,
        dict_mask: Tensor | None = None,
        dict_data=None,
    ) -> Tensor:
        src_embs = self.src_embed(src_nums)
        if dict_data is not None:
            for i, (lemmas, senses) in enumerate(dict_data):
                for (a, _), sense_spans in zip(lemmas, senses):
                    for c, d in sense_spans:
                        src_embs[i, c:d] = src_embs[i, a] + self.dpe_embed(src_nums[i, c:d])
        return self.encoder(src_embs, src_mask, dict_mask)

    def decode(
        self,
        src_encs: Tensor,
        tgt_nums: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        tgt_embs = self.tgt_embed(tgt_nums)
        return self.decoder(src_encs, tgt_embs, src_mask, tgt_mask)

    def forward(
        self,
        src_nums: Tensor,
        tgt_nums: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        dict_mask: Tensor | None = None,
        dict_data=None,
    ) -> Tensor:
        src_encs = self.encode(src_nums, src_mask, dict_mask, dict_data)
        tgt_encs = self.decode(src_encs, tgt_nums, src_mask, tgt_mask)
        return self.out_embed(tgt_encs, inverse=True)
