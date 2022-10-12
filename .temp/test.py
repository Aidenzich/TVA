#%%
import sys

sys.path.append("../")
from src.datasets.common import RecsysData
from src.configs.paths import DATA_PATH
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from src.models.VAECF.utils import split_matrix_by_mask, recall_calculate
from src.datasets.matrix_dset import MatrixDataset
import torch
import numpy as np

with open(DATA_PATH / "data_cls.pkl", "rb") as f:
    recsys_data = pickle.load(f)

valset = MatrixDataset(recsys_data.val_matrix)
val_loader = DataLoader(
    valset,
    batch_size=12,
    shuffle=False,
    pin_memory=True,
)
for i in val_loader:
    # print(i.shape)

    x, true_y, __ = split_matrix_by_mask(i)
    print(x.shape)
    print(true_y.shape)
    rand_tensor = torch.rand_like(true_y)
    seen = x != 0
    print(rand_tensor.shape)
    print(seen)
    rand_tensor[seen] = 0
    # print(x.shape)
    # print(true_y.shape)
    # top_idx = rand_tensor.topk(100, dim=1).indices
    print(recall_calculate(rand_tensor, true_y, k=10000))

    break
