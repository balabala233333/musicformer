import torch
from torch import nn, Tensor
from typing import Tuple, List

from .embedding import TransformerPositionalEncoding


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            n_class: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
            layer_norm_eps: float = 1e-05,
            batch_first: bool = True,
            norm_first: bool = False,
            blank_id: int = 0,
            device=None,
            **kwargs
    ):
        super().__init__()
        self.embedding = nn.Embedding(n_class, d_model)
        self.encoding = TransformerPositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        self.output_dim = d_model
        self.blank_id = blank_id
        self.device = device

    def forward(
            self,
            targets: Tensor,
            encoder_outputs: Tensor,
            hidden_state: Tensor = None,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:
        tgt_mask, tgt_padding_mask = self.create_mask(targets)
        tgt_mask, tgt_padding_mask = (
            tgt_mask.to(self.device),
            tgt_padding_mask.to(self.device),
        )

        embedded = self.embedding(targets)
        inputs = self.encoding(embedded)

        outputs = self.decoder(
            tgt=inputs,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        return outputs, hidden_state

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = torch.tril(torch.ones((sz, sz))) == 1
        mask = (
            mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, tgt: Tensor) -> Tuple[Tensor, Tensor]:
        seq_len = tgt.size(1)

        mask = self.generate_square_subsequent_mask(seq_len)

        padding_mask = tgt == self.blank_id
        return mask, padding_mask
