from turtle import shape
import numpy as np
from sklearn.metrics import f1_score
import torch


def split_matrix_by_mask(matrix):
    # print(matrix.shape)
    buy_idxs = np.nonzero(matrix)  # SHAPE (USER_ROW, ITEM_COL)
    buy_idxs = buy_idxs.transpose(0, 1)

    # print(buy_idxs.shape)
    random_items_idx = np.random.permutation(len(buy_idxs[0]))
    mask_num = int(len(random_items_idx) * 0.2)
    test_idx = random_items_idx[mask_num:]
    train_idx = random_items_idx[:mask_num]
    # print(test_idx.shape)
    # print(test_idx)
    masked_idx = (
        buy_idxs[0][test_idx],
        buy_idxs[1][test_idx],
    )

    non_masked_idx = (
        buy_idxs[0][train_idx],
        buy_idxs[1][train_idx],
    )

    masked_matrix = matrix.clone()
    masked_matrix[non_masked_idx] = 0

    non_masked_matrix = matrix.clone()
    non_masked_matrix[masked_idx] = 0

    return non_masked_matrix, masked_matrix, masked_idx


def recall_precision_f1_calculate(pred_tensor, true_tensor, k=100):
    true_idxs = true_tensor != 0
    pred_top_idx = pred_tensor.topk(k, dim=1).indices
    index_indicator = pred_top_idx == true_tensor.argmax(dim=1).unsqueeze(dim=1)
    row_indicator = index_indicator.any(dim=1)
    top_2_matches = torch.arange(len(row_indicator))[row_indicator]
    recall = len(top_2_matches) / len(true_idxs)
    precision = len(top_2_matches) / k
    f1_score = 2 * (precision*recall) / (precision + recall)
    return recall, precision, f1_score

