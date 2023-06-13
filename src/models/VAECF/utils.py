import numpy as np
import torch
from torch import Tensor
from typing import Tuple
from scipy.sparse import csr_matrix


def recall_precision_f1_calculate(
    pred_tensor: Tensor, true_tensor: Tensor, k: int = 100
) -> Tuple[float, float, float]:
    nonzero = torch.nonzero(true_tensor)
    pred_top_idx = pred_tensor.topk(k, dim=1).indices

    pred_in_true = torch.gather(true_tensor, axis=1, index=pred_top_idx)
    hit_num = len(torch.nonzero(pred_in_true))

    recall = hit_num / len(nonzero)
    precision = hit_num / (k * true_tensor.shape[0])

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return recall, precision, f1_score
