import torch


def rpf1_for_ks(scores: torch.tensor, labels: torch.tensor, ks):
    # BATCH, SAMPLE_SIZE + 1

    metrics = {}

    scores = scores
    labels = labels
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

        metrics["precision@%d" % k] = (
            (hits.sum(1) / torch.Tensor([k]).to(labels.device)).mean().cpu().item()
        )

        metrics["f1@%d" % k] = 0
        if (metrics["recall@%d" % k] + metrics["precision@%d" % k]) != 0:
            metrics["f1@%d" % k] = (
                2
                * metrics["recall@%d" % k]
                * metrics["precision@%d" % k]
                / (metrics["recall@%d" % k] + metrics["precision@%d" % k])
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


def ndcg(scores, labels, k):
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


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    scores = scores
    labels = labels
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
    scores = scores
    labels = labels
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
