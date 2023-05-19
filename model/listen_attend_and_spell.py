import os.path

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from model.encoder import Conformer
from model.decoder import TransformerDecoder

from data_process.vocabulary import TokensVocabulary, build_codec

from constant import TokenConfig


class LAS(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 n_class: int):
        super(LAS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.save_base_path = None
        self.out = nn.Linear(decoder.output_dim, n_class)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ):
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, hidden_state = self.decoder(targets, encoder_outputs)
        outputs = self.out(decoder_outputs)
        outputs = F.log_softmax(outputs, -1)
        return outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> str:

        encoder_output = encoder_output.unsqueeze(0)
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        targets = torch.IntTensor([vocabulary.start_of_tokens]).to(encoder_output.device)

        for i in range(max_length):

            targets = targets.unsqueeze(0)

            decoder_outputs, hidden_state = self.decoder(
                targets, encoder_output
            )
            outputs = F.log_softmax(self.out(decoder_outputs), -1)

            last_token = outputs.squeeze(0).argmax(-1)[-1]

            targets = torch.concat((targets.squeeze(0), last_token.unsqueeze(0)), -1)

            if last_token == vocabulary.end_of_tokens:
                break

        return targets

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> List[str]:
        encoder_outputs, _ = self.encoder(inputs, input_lengths)

        outputs = list()
        max_length = 1024

        for encoder_output in encoder_outputs:
            predict = self.decode(encoder_output, max_length)
            outputs.append(predict)

        return outputs


def build_conformer_listen_attend_and_spell_from_config():
    codec = build_codec()
    token_config = TokenConfig()

    conformer_encoder = Conformer(input_dim=token_config.encoder_input_dim, encoder_dim=token_config.encoder_input_dim,
                                  num_encoder_layers=token_config.num_encoder_layers,
                                  num_attention_heads=token_config.num_encoder_attention_heads)
    transformer_decoder = TransformerDecoder(n_class=codec.num_classes, d_model=token_config.decoder_input_dim,
                                             nhead=token_config.num_decoder_attention_heads,
                                             num_layers=token_config.num_decoder_layers, device=token_config.device)

    conformer_encoder = conformer_encoder.to(token_config.device)
    transformer_decoder = transformer_decoder.to(token_config.device)
    conformer_las = LAS(conformer_encoder, transformer_decoder, TokensVocabulary(codec.num_classes).vocabulary_size).to(
        token_config.device)

    print(conformer_las)
    conformer_las = conformer_las.to(token_config.device)
    conformer_las.save_base_path = token_config.checkpoint_path
    if os.path.exists(token_config.checkpoint_path):
        os.listdir(token_config.checkpoint_path)

    return conformer_las


def load_conformer_listen_attend_and_spell_from_checkpoint(path: str):
    codec = build_codec()
    token_config = TokenConfig()

    conformer_encoder = Conformer(input_dim=token_config.encoder_input_dim, encoder_dim=token_config.encoder_input_dim,
                                  num_encoder_layers=token_config.num_encoder_layers,
                                  num_attention_heads=token_config.num_encoder_attention_heads)
    transformer_decoder = TransformerDecoder(n_class=codec.num_classes, d_model=token_config.decoder_input_dim,
                                             nhead=token_config.num_decoder_attention_heads,
                                             num_layers=token_config.num_decoder_layers, device=token_config.device)

    conformer_encoder = conformer_encoder.to(token_config.device)
    transformer_decoder = transformer_decoder.to(token_config.device)

    conformer_las = LAS(conformer_encoder, transformer_decoder, 1517).to(token_config.device)

    state_dict = torch.load(path, map_location=token_config.device)
    conformer_las.load_state_dict(state_dict)
    conformer_las.save_base_path = token_config.checkpoint_path

    return conformer_las
