import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def generate_seqlen_group(recdata):
    seq_len_10 = []
    seq_len_20 = []
    seq_len_30 = []
    seq_len_40 = []
    seq_len_large = []

    for i in tqdm(recdata.users_seqs):
        seq_len = len(recdata.users_seqs[i])
        seq_len = seq_len - 1
        if seq_len <= 10:
            seq_len_10.append(i)
        elif seq_len <= 20:
            seq_len_20.append(i)
        elif seq_len <= 30:
            seq_len_30.append(i)
        elif seq_len <= 40:
            seq_len_40.append(i)
        else:
            seq_len_large.append(i)

    print("==" * 50)
    print(len(seq_len_10))
    print(len(seq_len_20))
    print(len(seq_len_30))
    print(len(seq_len_40))
    print(len(seq_len_large))
    print("==" * 50)

    return seq_len_10, seq_len_20, seq_len_30, seq_len_40, seq_len_large


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
