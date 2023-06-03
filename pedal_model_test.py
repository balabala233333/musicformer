import note_seq
import pandas as pd
import torch

from constant import TokenConfig, TEST_PEDAL_CHECKPOINT_PATH
from data_process import spectrograms
from data_process.datasets import build_maestrov2_dataset
from data_process.postprocess import trans_tokens_to_midi
from data_process.preprocess import test_midi_and_audio_to_pedal_tokens, test_midi_and_audio_to_note_pedal_tokens, \
    test_midi_and_audio_to_tokens
from data_process.spectrograms import SpectrogramConfig
from data_process.vocabulary import TokensVocabulary, build_codec
import tensorflow as tf

from evaluation.metrics import get_scores, get_pedal_score
from model.listen_attend_and_spell import load_conformer_listen_attend_and_spell_from_checkpoint

tf.config.set_visible_devices([], 'GPU')
config = build_maestrov2_dataset()

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
for pair in config.test_pairs:
    model = load_conformer_listen_attend_and_spell_from_checkpoint("/root/autodl-tmp/conformer/conformer_pedal_67")
    model.eval()
    for pair in config.test_pairs:
        token_config = TokenConfig()
        audio = spectrograms.read_wave(pair.audio_file_name, 16000)
        items = test_midi_and_audio_to_pedal_tokens(pair, SpectrogramConfig(), build_codec(), TokenConfig(),
                                              TokensVocabulary(build_codec().num_classes))
        for item in items:
            inputs = torch.unsqueeze(item.inputs, dim=0).to(token_config.device)
            input_lengths = torch.unsqueeze(item.input_lengths, dim=0).to(token_config.device)
            item.pred = model.recognize(inputs, input_lengths)[0]
        pred_ns = trans_tokens_to_midi(items, "res.midi")
        target_ns = note_seq.midi_file_to_note_sequence(
            pair.midi_file_name)
        res = get_scores(target_ns, pred_ns)
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
    df.to_excel("res.xlsx")
