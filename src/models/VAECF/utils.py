# from turtle import shape
import numpy as np
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
    nonzero = torch.nonzero(true_tensor)
    pred_top_idx = pred_tensor.topk(k, dim=1).indices

    pred_in_true = torch.gather(true_tensor, axis=1, index=pred_top_idx)
    hit_num = len(torch.nonzero(pred_in_true))

    # print(len(nonzero), hit_num)
    recall = hit_num / len(nonzero)
    precision = hit_num / (k * true_tensor.shape[0])
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1_score
