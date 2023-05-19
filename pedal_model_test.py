import note_seq

from constant import TokenConfig
from data_process.datasets import build_maestrov2_dataset
from data_process.postprocess import trans_tokens_to_midi
from data_process.preprocess import test_midi_and_audio_to_pedal_tokens
from data_process.spectrograms import SpectrogramConfig
from data_process.vocabulary import TokensVocabulary, build_codec
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
config = build_maestrov2_dataset()
for pair in config.test_pairs:
    print(note_seq.midi_file_to_note_sequence(pair.midi_file_name).control_changes)
    items = test_midi_and_audio_to_pedal_tokens(pair, SpectrogramConfig(), build_codec(), TokenConfig(),
                                          TokensVocabulary(build_codec().num_classes))

    for item in items:
        item.pred = item.targets
    pred_ns = trans_tokens_to_midi(items, "res.midi")
    print(pred_ns)