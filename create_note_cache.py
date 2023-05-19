import os

import torch

from constant import DATASET_NAME, TokenConfig
from data_process import spectrograms, datasets
from data_process.datasets import build_maestrov3_dataset, build_maestrov2_dataset, build_maestrov1_dataset
from data_process.preprocess import tokenize_example, split_data
from data_process.vocabulary import build_codec, TokensVocabulary


def create_cache():
    if DATASET_NAME == "maestro-v3.0.0":
        config = build_maestrov3_dataset()
    if DATASET_NAME == "maestro-v2.0.0":
        config = build_maestrov2_dataset()
    if DATASET_NAME == "maestro-v1.0.0":
        config = build_maestrov1_dataset()

    pairs = config.train_pairs
    token_config = TokenConfig()
    spectrogram_config = spectrograms.SpectrogramConfig()
    codec = build_codec()
    vocabulary = TokensVocabulary(codec.num_classes)

    chuck = []
    cnt = 0

    for pair in pairs:
        cache_path = os.path.join(pair.cache_data_path, pair.id + ".pt")
        if os.path.exists(cache_path):
            print(f"load cache {cache_path}")
            split_data_items = torch.load(cache_path)
        else:
            pair = datasets.trans_path_to_raw_data(pair)
            preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
            split_data_items = split_data(dataset=preprocessed_data, config=token_config,
                                          cache_path=pair.cache_path)

        for split_data_item in split_data_items:
            path = os.path.join(config.cache_split_data_path, str(cnt) + ".pt")
            torch.save(split_data_item, path)
            cnt = cnt + 1
    for pair in pairs:
        cache_path = os.path.join(pair.cache_data_path, pair.id + ".pt")
        if os.path.exists(cache_path):
            print(f"load cache {cache_path}")
            split_data_items = torch.load(cache_path)
        else:
            pair = datasets.trans_path_to_raw_data(pair)
            preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
            split_data_items = split_data(dataset=preprocessed_data, config=token_config,
                                          cache_path=pair.cache_path)

        for split_data_item in split_data_items:
            chuck.append(split_data_item)
            if len(chuck) % 1000 == 0:
                print(f"cached {len(chuck)} item")


if __name__ == '__main__':
    create_cache()
