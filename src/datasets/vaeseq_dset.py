import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random


class VAESequenceDataset(Dataset):
    def __init__(
        self,
        u2seq,
        u2timeseq,
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
        latent_factor=None,
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
        self.u2time_seq = u2timeseq
        self.latent_factor = latent_factor

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_latent_factor = self.latent_factor[index]
        user_vae_matrix = self.vae_matrix[index]
        user = self.users[index]
        seq = self.u2seq[user]
        time_seq = self.u2time_seq[user]
        time_interval_seq = [0] + [
            time_seq[i] - time_seq[i - 1] for i in range(1, len(time_seq))
        ]

        vae_seq = []

        if self.mode == "eval":
            answer = self.u2answer[user]
            negs = self.negative_samples[user]

            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)

            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            # time_seq
            time_seq = time_seq + [0]
            time_seq = time_seq[-self.max_len :]
            time_seq = [0] * padding_len + time_seq

            # time_interval_seq
            time_interval_seq = time_interval_seq + [0]
            time_interval_seq = time_interval_seq[-self.max_len :]
            time_interval_seq = [0] * padding_len + time_interval_seq

            for i in range(len(seq)):
                if seq[i] == 0:
                    vae_seq.append(0)
                elif seq[i] == self.mask_token:
                    vae_seq.append(0)
                else:
                    vae_seq.append(user_vae_matrix[seq[i]])

            return (
                torch.LongTensor(seq),  # user's sequence
                torch.FloatTensor(vae_seq),  # latent factor of user's sequence
                torch.FloatTensor(time_seq),  # timestamps of user's sequence
                torch.FloatTensor(
                    time_interval_seq
                ),  # time intervals of user's sequence
                torch.FloatTensor(user_latent_factor),  # latent factor
                torch.LongTensor(candidates),  # candidates from negative sampling
                torch.LongTensor(
                    labels
                ),  # labels from user's answer and negative samples
            )

        if self.mode == "train":
            tokens = []
            labels = []
            timestamps = []
            timestamps_interval = []
            for idx, s in enumerate(seq):
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                        timestamps.append(0)
                        timestamps_interval.append(0)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                        timestamps.append(0)
                        timestamps_interval.append(0)
                    else:
                        tokens.append(s)
                        timestamps.append(time_seq[idx])
                        timestamps_interval.append(time_interval_seq[idx])

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)
                    timestamps.append(time_seq[idx])
                    timestamps_interval.append(time_interval_seq[idx])

            tokens = tokens[-self.max_len :]
            labels = labels[-self.max_len :]
            timestamps = timestamps[-self.max_len :]
            timestamps_interval = timestamps_interval[-self.max_len :]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels
            timestamps = [0] * mask_len + timestamps
            timestamps_interval = [0] * mask_len + timestamps_interval

            # variational latent factor
            for i in range(len(tokens)):
                if tokens[i] == 0:
                    vae_seq.append(0)
                elif tokens[i] == self.mask_token:
                    vae_seq.append(0)
                else:
                    vae_seq.append(user_vae_matrix[tokens[i]])

            return (
                torch.LongTensor(tokens),  # masked user's sequence
                torch.FloatTensor(vae_seq),  # latent factor of masked user's sequence
                torch.FloatTensor(timestamps),  # masked user's timestamps
                torch.FloatTensor(timestamps_interval),  # masked user's time intervals
                torch.FloatTensor(user_latent_factor),  # latent factor
                torch.LongTensor(labels),  # labels for masked tokens
                torch.empty((0)),
            )

        if self.mode == "inference":
            candidates = self.negative_samples[user]
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            for i in range(len(seq)):
                if seq[i] == 0:
                    vae_seq.append(0)
                elif seq[i] == self.mask_token:
                    vae_seq.append(0)
                else:
                    vae_seq.append(user_vae_matrix[seq[i]])

            return (
                torch.LongTensor(seq),
                torch.FloatTensor(vae_seq),  # user's sequence
                torch.LongTensor(candidates),  # candidates from negative sampling
                torch.LongTensor([user]),
            )
