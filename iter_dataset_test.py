import math
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

class MyDataset(IterableDataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        p = pathlib.Path(self.path)
        self.file_list = list(p.glob('*.npy'))
        self.file_list_len = len(self.file_list)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.file_list_len
        else:
            per_worker = int(math.ceil(self.file_list_len / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = 0 + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.file_list_len)
        for i in range(iter_start, iter_end):
            arr = np.load(self.file_list[i])
            for j in range(arr.shape[0]):
                yield arr[j]


path = '260_1'
train_ds = MyDataset(path)
train_iter = DataLoader(train_ds, batch_size=64, num_workers=4)

res = 0
for i, sample in enumerate(train_iter):
    res += 1
    print(sample.shape)