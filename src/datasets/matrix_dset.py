from torch.utils.data import Dataset
import torch


class MatrixDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        buy_items = self.data[idx, :].A[0]
        return torch.tensor(buy_items, dtype=torch.float32)
