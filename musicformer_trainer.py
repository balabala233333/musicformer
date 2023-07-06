import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

from constant import MUSICFORMER_TARGETS_LENGTH, BATCH_SIZE, TokenConfig, NOTE_PEDAL_MODEL_EPOCHS, \
    NOTE_PEDAL_MODEL_SCHEDULER_WARMUP_RATIO, LEARNING_RATE, BETAS, EPS, WEIGHT_DECAY, CHECKPOINT_PATH, \
    MUSICFORMER_MODEL_SCHEDULER_WARMUP_RATIO, MUSICFORMER_MODEL_EPOCHS
from data_process.datasets import build_maestrov3_dataset
from data_process.vocabulary import build_codec
from data_set.data_set import PedalNoteDataset, TestDataset, PedalNoteTestset
from model.decoder import TransformerDecoder
from model.encoder import TransformerEncoder, Conformer
from model.musicformer import MusicFormer
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

torch.set_default_dtype(torch.float16)
def collate_fn(batch):
    inputs = []
    targets = []
    pedals = []
    input_lengths = []
    for input, input_length, pedal, pedal_length, target, target_length in batch:
        inputs.append(input)
        targets.append(target)
        pedals.append(pedal)
        input_lengths.append(input_length)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(dtype=torch.int)
    pedals = torch.nn.utils.rnn.pad_sequence(pedals, batch_first=True).to(dtype=torch.int)
    pedals = torch.nn.functional.pad(pedals, (0, MUSICFORMER_TARGETS_LENGTH - pedals.shape[1]), mode='constant',
                                     value=0)
    input_lengths = torch.stack(input_lengths, dim=0)
    target_lengths = torch.IntTensor([s.size(0) for s in targets])
    pedal_lengths = torch.IntTensor([s.size(0) for s in pedals])
    return inputs, input_lengths, pedals, pedal_lengths, targets, target_lengths


token_config = TokenConfig()
pedal_encoder = TransformerEncoder(input_dim=token_config.encoder_input_dim, d_model=token_config.encoder_input_dim,
                                   nhead=token_config.num_encoder_attention_heads,
                                   num_layers=token_config.num_encoder_layers)
wav_encoder = Conformer(input_dim=token_config.encoder_input_dim, encoder_dim=token_config.encoder_input_dim,
                        num_encoder_layers=token_config.num_encoder_layers,
                        num_attention_heads=token_config.num_encoder_attention_heads)
codec = build_codec()
transformer_decoder = TransformerDecoder(n_class=codec.num_classes, d_model=token_config.decoder_input_dim,
                                             nhead=token_config.num_decoder_attention_heads,
                                             num_layers=token_config.num_decoder_layers)
musicformer = MusicFormer(pedal_encoder, wav_encoder, transformer_decoder)
musicformer = nn.DataParallel(musicformer,device_ids=[0])


config = build_maestrov3_dataset()
dataset = PedalNoteDataset(config)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, prefetch_factor=2)

testset = PedalNoteTestset(config)
test_loader = DataLoader(testset, batch_size=8, collate_fn=collate_fn, num_workers=4, prefetch_factor=2)
epochs = MUSICFORMER_MODEL_EPOCHS
num_training_steps = epochs * len(data_loader)
scheduler_warmup_ratio = MUSICFORMER_MODEL_SCHEDULER_WARMUP_RATIO
num_warmup_steps = int(num_training_steps * scheduler_warmup_ratio)
criterion = CrossEntropyLoss()
optimizer = AdamW(musicformer.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS,
                  weight_decay=WEIGHT_DECAY)
scheduler = get_scheduler('polynomial',
                          optimizer,
                          num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps,
                          )
result = []
for i in range(MUSICFORMER_MODEL_EPOCHS):
    musicformer.train()
    cnt = 0
    for wav_inputs, wav_inputs_length, pedal_inputs, pedal_inputs_length, targets, targets_length in data_loader:
        optimizer.zero_grad()
        targets_in = targets[:, :-1]
        targets_out = targets[:, 1:]
        res = musicformer(pedal_inputs, pedal_inputs_length, wav_inputs, wav_inputs_length,
                          targets_in)
        bz, t, _ = res.size()
        loss = criterion(torch.squeeze(res.cpu(), 0).view(bz * t, -1), torch.squeeze(targets_out.long(), 0).view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        tokens = res.argmax(-1)
        cnt = cnt + 1
        if cnt % 100 == 0:
            print("========================train{}:loss:{},lr:{}==========================".format(i, loss,
                                                                                                   scheduler.get_last_lr()))
    musicformer.eval()
    tot = 0
    cnt = 0
    for wav_inputs, wav_inputs_length, pedal_inputs, pedal_inputs_length, targets, targets_length in data_loader:
        targets_in = targets[:, :-1]
        targets_out = targets[:, 1:]
        res = musicformer(pedal_inputs, pedal_inputs_length, wav_inputs, wav_inputs_length,
                          targets_in)
        bz, t, _ = res.size()
        loss = criterion(torch.squeeze(res.cpu(), 0).view(bz * t, -1), torch.squeeze(targets_out.long(), 0).view(-1))
        tot = tot + loss.item()
        cnt = cnt + 1
        if cnt % 100 == 0:
            print("========================test{}:loss:{}==========================".format(i, loss))
    result.append(tot / cnt)
    print(result)
    torch.save(musicformer.state_dict(),
               os.path.join(CHECKPOINT_PATH, "musicformer_{}".format(i)))