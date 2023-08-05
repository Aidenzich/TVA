import torch
from torch import Tensor
from typing import Dict, List
import numpy as np

METRICS_KS = [1, 5, 10, 20]


def ndcg(scores: Tensor, labels: Tensor, k: int = 10) -> Tensor:
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(
    scores: Tensor, labels: Tensor, ks: List[int]
) -> Dict[str, float]:
    metrics = {}
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics["recall@%d" % k] = (
            (
                hits.sum(1)
                / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
            )
            .mean()
            .cpu()
            .item()
        )

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(
            dcg.device
        )
        ndcg = (dcg / idcg).mean()
        metrics["ndcg@%d" % k] = ndcg.cpu().item()

    return metrics


def recall_precision_f1_calculate(scores, labels, k):
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    recall = (
        (
            hit.sum(1).float()
            / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())
        )
        .mean()
        .cpu()
        .item()
    )

    precision = (hit.sum(1).float() / torch.Tensor([k])).mean().cpu().item()
    f1_score = 0
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return recall, precision, f1_score


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.0


def ndcg_at_k(pred_list, true_list, k):
    pred_labels = [1 if i in true_list else 0 for i in pred_list]
    true_labels = [1 if i in true_list else 0 for i in true_list]

    idcg = dcg_at_k(sorted(true_labels, reverse=True), k)
    dcg = dcg_at_k(pred_labels, k)
    return dcg / idcg if idcg > 0.0 else 0.0


def recall_at_k(pred_list, true_list, k):
    # Get top-k items
    top_k_items = pred_list[:k]

    # Check if true_item is among top-k
    relevant_count = 0
    for true_item in true_list:
        if true_item in set(top_k_items):
            relevant_count += 1

    # Avoid division by zero
    if len(true_list) == 0:
        return 0

    return relevant_count / len(true_list)
