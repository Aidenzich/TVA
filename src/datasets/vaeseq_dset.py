import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
import random
from pydantic import BaseModel, ValidationError


class VAESequences(BaseModel):
    userwise_latent_factor: Optional[torch.FloatTensor]
    time_interval_seq: Optional[torch.FloatTensor]
    candidates: Optional[torch.LongTensor]
    time_seq: Optional[torch.FloatTensor]
    item_seq: Optional[torch.LongTensor]
    vae_seq: Optional[torch.LongTensor]
    labels: Optional[torch.LongTensor]

    class Config:
        arbitrary_types_allowed = True


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
        latent_factor=None,
    ) -> None:

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
        self.u2time_seq = u2timeseq
        self.latent_factor = latent_factor

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        user_latent_factor = self.latent_factor[index]
        user = self.users[index]
        seq = self.u2seq[user]
        time_seq = self.u2time_seq[user]
        time_interval_seq = [0] + [
            time_seq[i] - time_seq[i - 1] for i in range(1, len(time_seq))
        ]

        vae_seq = []

        if self.mode == "eval":

            # candidates: candidates from negative sampling
            answer = self.u2answer[user]
            negs = self.negative_samples[user]
            candidates = answer + negs

            # labels: labels from user's answer and negative samples
            labels = [1] * len(answer) + [0] * len(negs)

            # seq: user's item sequence
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            # time_seq: timestamps of user's sequence
            time_seq = time_seq + [0]
            time_seq = time_seq[-self.max_len :]
            time_seq = [0] * padding_len + time_seq

            # time_interval_seq: time intervals of user's sequence
            time_interval_seq = time_interval_seq + [0]
            time_interval_seq = time_interval_seq[-self.max_len :]
            time_interval_seq = [0] * padding_len + time_interval_seq

            data = {
                "item_seq": torch.LongTensor(seq),
                "time_seq": torch.FloatTensor(time_seq),
                "time_interval_seq": torch.FloatTensor(time_interval_seq),
                "userwise_latent_factor": torch.FloatTensor(user_latent_factor),
                "candidates": torch.LongTensor(candidates),
                "labels": torch.LongTensor(labels),
            }

            return VAESequences(**data).dict(exclude_none=True)

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

            data = {
                "item_seq": torch.LongTensor(tokens),
                "time_seq": torch.FloatTensor(timestamps),
                "time_interval_seq": torch.FloatTensor(timestamps_interval),
                "userwise_latent_factor": torch.FloatTensor(user_latent_factor),
                "candidates": torch.empty((0)),
                "labels": torch.LongTensor(labels),
            }

            return VAESequences(**data).dict(exclude_none=True)

        if self.mode == "inference":
            candidates = self.negative_samples[user]
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            data = {
                "item_seq": torch.LongTensor(seq),
                "vae_seq": torch.empty((0)),
                "time_seq": torch.FloatTensor(timestamps),
                "time_interval_seq": torch.FloatTensor(timestamps_interval),
                "userwise_latent_factor": torch.FloatTensor(user_latent_factor),
                "candidates": torch.LongTensor(candidates),
            }

            return data

            return (
                torch.LongTensor(seq),
                torch.FloatTensor(vae_seq),  # user's sequence
                torch.LongTensor(candidates),  # candidates from negative sampling
                torch.LongTensor([user]),
            )
