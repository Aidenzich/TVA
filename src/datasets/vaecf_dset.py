import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Any
from src.datasets.base import RecsysData
import random


class VAECFDataset(Dataset):
    def __init__(self, data, split_type, u2val=None, mode: str = "train") -> None:
        self.data = data
        self.u2val = u2val
        self.mode = mode
        self.split_type = split_type

    def __len__(self) -> int:
        return self.data.shape[0]

    def _get_loo_eval(self, idx) -> bool:
        rdata = self.data[idx, :].A[0]

        return {
            "matrix": torch.tensor(
                rdata,
                dtype=torch.float32,
            ),
            "validate_data": self.u2val[idx] if self.u2val else None,
        }

    def _get_matrix_tensor(self, idx) -> bool:
        return {
            "matrix": torch.tensor(
                self.data[idx, :].A[0],
                dtype=torch.float32,
            )
        }

    def __getitem__(self, idx) -> Tensor:
        if self.split_type == "loo":
            if self.mode == "train":
                return self._get_matrix_tensor(idx)
            else:
                return self._get_loo_eval(idx)
        else:
            return self._get_matrix_tensor(idx)


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


def _split_random_matrix_by_user(recdata: RecsysData) -> Tuple[Any, Any, Any]:
    """
    Splitting matrix by random shuffle user for train, testing and validation
    """

    print("*Splitting matrix by random shuffle user for train, testing and validation*")

    # split to train val and test
    users = list(recdata.u2cat.values())

    random.shuffle(users)
    train_num = int(len(users) * 0.98)
    test_num = int(len(users) * 0.01)

    train_users = users[:train_num]
    test_users = users[-test_num:]
    val_users = users[train_num:-test_num]

    train_matrix = recdata.matrix[train_users, :]
    test_matrix = recdata.matrix[test_users, :]
    val_matrix = recdata.matrix[val_users, :]

    return train_matrix, test_matrix, val_matrix


def split_random_matrix_by_item(recdata: RecsysData):
    """
    Splitting matrix by random shuffle user for train, testing and validation
    """

    print("Splitting matrix by random shuffle user for train, testing and validation")

    # split to train val and test
    items = list(recdata.i2cat.values())

    random.shuffle(items)
    train_num = int(len(items) * 0.98)
    test_num = int(len(items) * 0.01)
    # val_num = len(users) - train_num - test_num
    # print(len(users[:train_num]))

    train_users = items[:train_num]
    test_users = items[-test_num:]
    val_users = items[train_num:-test_num]

    matrix = recdata.matrix.transpose()

    train_matrix = matrix[train_users, :]
    test_matrix = matrix[test_users, :]
    val_matrix = matrix[val_users, :]

    return train_matrix, test_matrix, val_matrix
