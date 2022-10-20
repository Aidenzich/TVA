import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random


class SequenceDataset(Dataset):
    def __init__(
        self,
        u2seq,
        max_len: int,
        mask_token,
        mode="train",  # train, eval, inference
        # for train
        num_items=0,
        mask_prob=0,
        seed=0,
        # for eval
        negative_samples=None,
        u2answer=None,
        vae_matrix=None,
    ):

        if mode == "eval":
            if negative_samples is None or u2answer is None:
                raise ValueError("negative_samples and u2answer must be provided")
        if mode == "train":
            if num_items == 0 or mask_prob == 0:
                raise ValueError("num_items, mask_prob must be provided")

        self.mode = mode
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = random.Random(seed)
        self.negative_samples = negative_samples
        self.u2answer = u2answer
        self.vae_matrix = vae_matrix

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_vae_matrix = self.vae_matrix[index]
        user = self.users[index]
        seq = self.u2seq[user]

        vae_seq = []
        for i in self.u2seq[user]:
            vae_seq.append(user_vae_matrix[i])

        if self.mode == "eval":
            answer = self.u2answer[user]
            negs = self.negative_samples[user]

            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)

            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            return (
                torch.LongTensor(seq),  # user's sequence
                torch.LongTensor(candidates),  # candidates from negative sampling
                torch.LongTensor(
                    labels
                ),  # labels from user's answer and negative samples
            )

        if self.mode == "train":
            tokens = []
            labels = []
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            tokens = tokens[-self.max_len :]
            labels = labels[-self.max_len :]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

            return (
                torch.LongTensor(tokens),  # masked user's sequence
                torch.LongTensor(labels),  # labels for masked tokens
                torch.empty((0)),
            )

        if self.mode == "inference":
            candidates = self.negative_samples[user]
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            return (
                torch.LongTensor(seq),  # user's sequence
                torch.LongTensor(candidates),  # candidates from negative sampling
                torch.LongTensor([user]),
            )
