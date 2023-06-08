from torch.utils.data import Dataset
import torch
import numpy as np


class VAECFDataset(Dataset):
    def __init__(self, data, wise="row"):
        self.data = data
        self.wise = wise

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # udata = self.data[idx, :]
        # udata.data = np.ones(len(udata.data))
        rdata = self.data[idx, :].A[0]

        return torch.tensor(rdata, dtype=torch.float32)
