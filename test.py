import os

import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler

from constant import WEIGHT_DECAY, LEARNING_RATE, BETAS, EPS, NOTE_MODEL_SCHEDULER_WARMUP_RATIO, NOTE_MODEL_EPOCHS, \
    BATCH_SIZE, TokenConfig, BLANK_ID, CHECKPOINT_PATH
from data_process.datasets import build_maestrov3_dataset
from data_set.data_set import TrainDataset, collate_fn, RNNTDataset
import tensorflow as tf
import torchaudio.transforms as T

from model.ConformerTransducer import build_conformer_transducer

tf.config.set_visible_devices([], 'GPU')

token_config = TokenConfig()
config = build_maestrov3_dataset()
dataset = TrainDataset(config, use_cache=True)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, prefetch_factor=2)
epochs = NOTE_MODEL_EPOCHS
num_training_steps = epochs * len(data_loader)
scheduler_warmup_ratio = NOTE_MODEL_SCHEDULER_WARMUP_RATIO
num_warmup_steps = int(num_training_steps * scheduler_warmup_ratio)

model = build_conformer_transducer()
# model = load_conformer_listen_attend_and_spell_from_checkpoint("/root/autodl-tmp/conformer/conformer_las_46")
criterion = T.RNNTLoss(blank=BLANK_ID)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS,
                  weight_decay=WEIGHT_DECAY)
scheduler = get_scheduler('polynomial',
                          optimizer,
                          num_warmup_steps=num_warmup_steps,
                          num_training_steps=num_training_steps,
                          )
cnt = 0
for i in range(epochs):
    for inputs, input_lengths, targets, target_lengths in data_loader:
        inputs = inputs.to(token_config.device)
        input_lengths = input_lengths.to(token_config.device)
        target_lengths = target_lengths.to(token_config.device)
        zeros = torch.zeros((targets.size(0), 1)).to(device=token_config.device)
        compute_targets = torch.cat((zeros, targets), dim=1).to(
            device=token_config.device, dtype=torch.int
        )
        compute_target_lengths = (target_lengths + 1).to(device=token_config.device)
        res,res_length = model(inputs, input_lengths, compute_targets, compute_target_lengths)
        res_length = torch.squeeze(res_length,dim=1).int()
        # print(targets,compute_targets)
        print("test1")
        loss = criterion(res,targets,res_length.int(), target_lengths.int())
        print("test2")
        loss.backward()
        optimizer.step()
        scheduler.step()
        cnt = cnt + 1
        # output, output_lengths = torch.ops.torchaudio.rnnt_decode(res, res_length)
        # print(output)
        # if cnt % 100 == 0:
        print("========================train{}:loss:{},lr:{}==========================".format(i, loss,scheduler.get_last_lr()))
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_PATH, "rnnt_{}".format(i)))