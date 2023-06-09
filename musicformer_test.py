import note_seq
import pandas as pd
import tensorflow as tf
import torch

from constant import TokenConfig
from data_process import spectrograms
from data_process.datasets import build_maestrov3_dataset
from data_process.postprocess import trans_tokens_to_midi
from data_process.preprocess import test_midi_and_audio_to_tokens
from data_process.spectrograms import SpectrogramConfig
from data_process.vocabulary import TokensVocabulary, build_codec
from evaluation.metrics import get_scores, get_pedal_score
from model.listen_attend_and_spell import build_conformer_ctc_from_checkpoint

tf.config.set_visible_devices([], 'GPU')
config = build_maestrov3_dataset()

df = pd.DataFrame()
name = []
mir_eval_onset_presion = []
mir_eval_onset_recall = []
mir_eval_onset_f1 = []
mir_eval_onset_offset_presion = []
mir_eval_onset_offset_recall = []
mir_eval_onset_offset_f1 = []
frame_precision = []
frame_recall = []
frame_f1 = []
mir_eval_vel_f1 = []
mir_eval_vel_recall = []
mir_eval_vel_presion = []
pedal_onset_f1 = []
pedal_onset_recall = []
pedal_onset_presion = []
pedal_onset_offset_f1 = []
pedal_onset_offset_recall = []
pedal_onset_offset_presion = []
model = build_conformer_ctc_from_checkpoint("/root/autodl-tmp/conformer/rnnt_4")
model.eval()
for pair in config.test_pairs:
    token_config = TokenConfig()
    audio = spectrograms.read_wave(pair.audio_file_name, 16000)
    items = test_midi_and_audio_to_tokens(pair, SpectrogramConfig(), build_codec(), TokenConfig(),
                                          TokensVocabulary(build_codec().num_classes))
    cnt = 0
    for item in items:
        inputs = torch.unsqueeze(item.inputs, dim=0).to(token_config.device)
        input_lengths = torch.unsqueeze(item.input_lengths, dim=0).to(token_config.device)
        item.pred = model.recognize(inputs, input_lengths)
        print(item.pred[0])
        print(cnt)
        cnt = cnt+1
    pred_ns = trans_tokens_to_midi(items, "res.midi")
    target_ns = note_seq.midi_file_to_note_sequence(
        pair.midi_file_name)
    res = get_scores(target_ns, pred_ns)
    print(res)
    mir_eval_onset_presion.append(res["onset_score"].precision_score)
    mir_eval_onset_recall.append(res["onset_score"].recall_score)
    mir_eval_onset_f1.append(res["onset_score"].f1_score)
    mir_eval_onset_offset_presion.append(res["onset_offset_score"].precision_score)
    mir_eval_onset_offset_recall.append(res["onset_offset_score"].recall_score)
    mir_eval_onset_offset_f1.append(res["onset_offset_score"].f1_score)
    mir_eval_onset_offset_presion.append(res["onset_offset_velocity_score"].precision_score)
    mir_eval_onset_offset_recall.append(res["onset_offset_velocity_score"].recall_score)
    mir_eval_onset_offset_f1.append(res["onset_offset_velocity_score"].f1_score)
    frame_precision.append(res["frame_score"].precision_score)
    frame_recall.append(res["frame_score"].recall_score)
    frame_f1.append(res["frame_score"].f1_score)
    res = get_pedal_score(target_ns, pred_ns)
    pedal_onset_presion.append(res["onset_score"].precision_score)
    pedal_onset_recall.append(res["onset_score"].recall_score)
    pedal_onset_f1.append(res["onset_score"].f1_score)
    pedal_onset_offset_f1.append(res["onset_offset_score"].f1_score)
    pedal_onset_offset_presion.append(res["onset_offset_score"].precision_score)
    pedal_onset_offset_recall.append(res["onset_offset_score"].recall_score)
    print(res)
df["name"] = name
df["pedal_onset_presion"] = pedal_onset_presion
df["pedal_onset_recall"] = pedal_onset_recall
df["pedal_onset_f1"] = pedal_onset_f1
df["pedal_onset_offset_f1"] = pedal_onset_offset_f1
df["pedal_onset_offset_presion"] = pedal_onset_offset_presion
df["pedal_onset_offset_recall"] = pedal_onset_offset_recall
df["mir_eval_onset_presion"] = mir_eval_onset_presion
df["mir_eval_onset_recall"] = mir_eval_onset_recall
df["mir_eval_onset_f1"] = mir_eval_onset_f1
df["mir_eval_onset_offset_presion"] = mir_eval_onset_offset_presion
df["mir_eval_onset_offset_recall"] = mir_eval_onset_offset_recall
df["mir_eval_onset_offset_f1"] = mir_eval_onset_offset_f1
df["mir_eval_vel_f1"] = mir_eval_vel_f1
df["mir_eval_vel_recall"] = mir_eval_vel_recall
df["mir_eval_vel_presion"] = mir_eval_vel_presion
df["frame_f1"] = frame_f1
df["frame_recall"] = frame_recall
df["frame_precision"] = frame_precision
df.to_excel("res.xlsx")
print("onset_precision:", sum(mir_eval_onset_presion) / len(mir_eval_onset_presion))
print("onset_recall:", sum(mir_eval_onset_recall) / len(mir_eval_onset_recall))
print("onset_f1:", sum(mir_eval_onset_f1) / len(mir_eval_onset_f1))
print("onset_offset_precision:", sum(mir_eval_onset_offset_presion) / len(mir_eval_onset_offset_presion))
print("onset_offset_recall:", sum(mir_eval_onset_offset_recall) / len(mir_eval_onset_offset_recall))
print("onset_offset_f1:", sum(mir_eval_onset_offset_f1) / len(mir_eval_onset_offset_f1))
print("onset_offset_velocity_precision:", sum(mir_eval_vel_presion) / len(mir_eval_vel_presion))
print("onset_offset_velocity_recall:", sum(mir_eval_vel_recall) / len(mir_eval_vel_recall))
print("onset_offset_velocity_f1:", sum(mir_eval_vel_f1) / len(mir_eval_vel_f1))
print("frame_precision:", sum(frame_precision) / len(frame_precision))
print("frame_recall:", sum(frame_recall) / len(frame_recall))
print("frame_f1:", sum(frame_f1) / len(frame_f1))

