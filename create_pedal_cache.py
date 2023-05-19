from torch.utils.data import DataLoader

from data_process.datasets import build_maestrov3_dataset
from data_set.data_set import PedalDataset, collate_fn

dataset = PedalDataset(build_maestrov3_dataset())
data_loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=4, prefetch_factor=1,
                             shuffle=True)
for k,v in data_loader:
    print(k,v)