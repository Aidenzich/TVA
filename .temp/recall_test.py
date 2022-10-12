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


pred = torch.tensor(
    [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ]
)

true = torch.tensor(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
    ]
)


recall_calculate(pred, true, k=2)

# %%
