import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from constant import NOTE_MODEL_EPOCHS, NOTE_MODEL_SCHEDULER_WARMUP_RATIO, TokenConfig, CHECKPOINT_PATH, LEARNING_RATE, \
    BETAS, EPS, WEIGHT_DECAY, PEDAL_MODEL_EPOCHS, PEDAL_MODEL_SCHEDULER_WARMUP_RATIO
from data_process.datasets import build_maestrov3_dataset
from data_set.data_set import collate_fn, PedalDataset, PedalTestDataset
from model.listen_attend_and_spell import build_conformer_listen_attend_and_spell_from_config
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

def pedal_model_train_step():
    token_config = TokenConfig()
    config = build_maestrov3_dataset()
    dataset = PedalDataset(config,use_cache=True)
    testset = PedalTestDataset(config,use_cache=True)
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=4, prefetch_factor=1,
                             shuffle=True)
    test_loader = DataLoader(testset, batch_size=8, collate_fn=collate_fn, num_workers=4, prefetch_factor=1,
                             shuffle=True)
    epochs = PEDAL_MODEL_EPOCHS
    num_training_steps = epochs * len(data_loader)
    scheduler_warmup_ratio = PEDAL_MODEL_SCHEDULER_WARMUP_RATIO
    num_warmup_steps = int(num_training_steps * scheduler_warmup_ratio)

    model = build_conformer_listen_attend_and_spell_from_config()
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS,
                      weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler('polynomial',
                              optimizer,
                              num_warmup_steps=num_warmup_steps,
                              num_training_steps=num_training_steps,
                              )
    result = []
    for i in range(epochs):
        for inputs, input_lengths, targets, target_lengths in data_loader:
            print(torch.max(targets))
            optimizer.zero_grad()
            targets_in = targets[:, :-1].to(token_config.device)
            targets_out = targets[:, 1:].to(token_config.device)
            inputs = inputs.to(token_config.device)
            input_lengths = input_lengths.to(token_config.device)
            target_lengths = target_lengths.to(token_config.device)
            res = model(inputs, input_lengths, targets_in, target_lengths)
            bz, t, _ = res.size()
            loss = criterion(torch.squeeze(res, 0).view(bz * t, -1), torch.squeeze(targets_out.long(), 0).view(-1))
            scheduler.get_lr()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens = res.argmax(-1)[-1]
            print("========================train{}:loss:{},lr:{}==========================".format(i, loss,
                                                                                                   scheduler.get_lr()))
            print(tokens)
        tot = 0
        cnt = 0
        for inputs, input_lengths, targets, target_lengths in test_loader:
            model.eval()
            targets_in = targets[:, :-1].to(token_config.device)
            targets_out = targets[:, 1:].to(token_config.device)
            inputs = inputs.to(token_config.device)
            input_lengths = input_lengths.to(token_config.device)
            target_lengths = target_lengths.to(token_config.device)
            res = model(inputs, input_lengths, targets_in, target_lengths)
            bz, t, _ = res.size()
            loss = criterion(torch.squeeze(res, 0).view(bz * t, -1), torch.squeeze(targets_out.long(), 0).view(-1))
            # loss.backward()
            # tokens = res.argmax(-1)[-1]
            tot = tot + loss.item()
            cnt = cnt + 1
            # print(tot,cnt,tot/cnt)

            print("========================test{}:loss:{}==========================".format(i, loss))
            # print(tokens)
        result.append(tot / cnt)
        print(result)
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_PATH, "conformer_pedal_{}".format(i)))


if __name__ == '__main__':
    pedal_model_train_step()
