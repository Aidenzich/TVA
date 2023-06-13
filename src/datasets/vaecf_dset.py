import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


class VAECFDataset(Dataset):
    def __init__(self, data, wise="row") -> None:
        self.data = data
        self.wise = wise

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> Tensor:
        rdata = self.data[idx, :].A[0]

        return torch.tensor(rdata, dtype=torch.float32)


def split_matrix_by_mask(matrix: csr_matrix) -> Tuple[Tensor, Tensor, Tuple[int, int]]:
    """
    This function splits the matrix into two parts, one for training and one for testing.
    """
    buy_idxs = np.nonzero(matrix)  # USER_NUM, ITEM_NUM
    buy_idxs = buy_idxs.transpose(0, 1)

    # Randomly shuffle the indicess
    random_items_idx = np.random.permutation(len(buy_idxs[0]))
    # The last 20% of the data is used for testing
    mask_num = int(len(random_items_idx) * 0.2)

    # The first 80% of the data is used for training
    train_idx = random_items_idx[:mask_num]
    # The last 20% of the data is used for testing
    test_idx = random_items_idx[mask_num:]

    masked_idx = (
        buy_idxs[0][test_idx],
        buy_idxs[1][test_idx],
    )

    non_masked_idx = (
        buy_idxs[0][train_idx],
        buy_idxs[1][train_idx],
    )

    masked_matrix, non_masked_matrix = matrix.clone(), matrix.clone()
    masked_matrix[non_masked_idx] = 0
    non_masked_matrix[masked_idx] = 0

    return non_masked_matrix, masked_matrix, masked_idx
