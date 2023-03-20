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
        num_items=0,
        # Train
        mask_prob=0,
        seed=0,
        # Evaluate
        u2answer=None,
    ) -> None:

        if mode == "eval":
            if u2answer is None:
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
        # self.negative_samples = negative_samples
        self.u2answer = u2answer

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[index]
        seq = self.u2seq[user]

        if self.mode == "eval":
            interacted = list(set(seq))
            interacted += self.u2answer[user]
            answer = self.u2answer[user]

            negs = [x for x in range(1, self.num_items + 1) if x not in interacted]

            # if negs is not enough, pad with 0
            if len(negs) < self.num_items:
                negs += [0] * (self.num_items - len(negs))

            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)

            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            return (
                # user's sequence
                torch.LongTensor(seq),
                # candidates from negative sampling
                torch.LongTensor(candidates),
                # labels from user's answer and negative samples
                torch.LongTensor(labels),
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
                # masked user's sequence
                torch.LongTensor(tokens),
                # labels for masked tokens
                torch.LongTensor(labels),
                torch.empty((0)),
            )

        if self.mode == "inference":
            candidates = [x for x in range(1, self.num_items + 1)]
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            return (
                # user's sequence
                torch.LongTensor(seq),
                # candidates from negative sampling
                torch.LongTensor(candidates),
                torch.LongTensor([user]),
            )
