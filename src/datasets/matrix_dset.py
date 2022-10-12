from torch.utils.data import Dataset
import torch
import numpy as np


class MatrixDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        udata = self.data[idx, :]
        udata.data = np.ones(len(udata.data))
        udata = self.data[idx, :].A[0]

        return torch.tensor(udata, dtype=torch.float32)
