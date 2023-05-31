import os

import note_seq
import torch
from torch.utils.data import Dataset

from constant import TokenConfig
from data_process import datasets, spectrograms, tokens_encoding
from data_process.preprocess import tokenize_example, split_data, select_random_chunk, compute_spectrogram, \
    trans_preprocess_data_item_to_train_data, test_midi_and_audio_to_tokens, tokenize_pedal, tokenize_note_pedal
from data_process.tokens_decoding import NoteDecodingState
from data_process.vocabulary import TokensVocabulary, build_codec


class TrainDataset(Dataset):
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_data_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache
        for pair in pairs:
            cache_path = os.path.join(config.cache_data_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items,cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


class TestDataset(Dataset):
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.test_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_data_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache
        for pair in pairs:
            cache_path = os.path.join(config.cache_data_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items,cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


class PedalDataset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache

        for pair in pairs:
            cache_path = os.path.join(config.cache_pedal_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_pedal(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


class PedalTestDataset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.test_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache

        for pair in pairs:
            cache_path = os.path.join(config.cache_pedal_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_pedal(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths

class NotePedalDataset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache

        for pair in pairs:
            cache_path = os.path.join(config.cache_note_pedal_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_note_pedal(pair=pair, spectrogram_config=spectrogram_config,
                                                        codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items,cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths




def collate_fn(batch):
    inputs = []
    targets = []
    input_lengths = []
    for input, input_length, target, target_length in batch:
        inputs.append(input)
        targets.append(target)
        input_lengths.append(input_length)
    inputs = torch.stack(inputs, dim=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(dtype=torch.int)
    input_lengths = torch.stack(input_lengths, dim=0)
    target_lengths = torch.IntTensor([s.size(0) for s in targets])
    return inputs, input_lengths, targets, target_lengths
