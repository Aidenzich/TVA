import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random
import numpy as np


class SASRecDataset(Dataset):
    def __init__(
        self,
        u2seq,
        max_len: int,
        mask_token,
        mode="train",  # train, eval, inference
        num_items=0,
        # Train parameters
        seed=0,
        negative_samples=None,
        # Eval parameters
        u2answer=None,
    ) -> None:

        if mode == "eval":
            if negative_samples is None:
                raise ValueError("negative_samples and u2answer must be provided")
        if mode == "train":
            if num_items == 0:
                raise ValueError("num_items must be provided")

        self.mode = mode
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = random.Random(seed)
        self.negative_samples = negative_samples
        self.u2answer = u2answer

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[index]
        seq = self.u2seq[user]

        train_seq, pos_label, neg_label = (
            np.zeros([self.max_len], dtype=np.int32) for i in range(3)
        )

        nxt_item = seq[-1]
        idx = self.max_len - 1

        for i in reversed(seq[:-1]):
            train_seq[idx] = i
            pos_label[idx] = nxt_item
            if nxt_item != 0:
                neg_label[idx] = random_neq(1, self.num_items + 1, set(seq))

            nxt_item = i
            idx -= 1
            if idx == -1:
                break

        if self.mode == "train":
            # (train_seq, pos_label, neg_label)
            return {
                "train_seq": torch.from_numpy(train_seq),
                "pos_label": torch.from_numpy(pos_label),
                "neg_label": torch.from_numpy(neg_label),
            }

        if self.mode == "eval":
            answer = self.u2answer[user]
            negs = self.negative_samples[user]
            candidates = answer + negs

            labels = [1] * len(answer) + [0] * len(negs)

            return {
                "candidates": torch.from_numpy(np.array(candidates)),
                "train_seq": torch.from_numpy(train_seq),
                "labels": torch.from_numpy(np.array(labels)),
            }


def random_neq(left, right, true_items):
    t = np.random.randint(left, right)
    while t in true_items:
        t = np.random.randint(left, right)
    return t
