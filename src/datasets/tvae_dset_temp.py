import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
import random
from pydantic import BaseModel, ValidationError
import numpy as np
import datetime


class TVASequences(BaseModel):
    time_seq: Optional[torch.FloatTensor]
    time_interval_seq: Optional[torch.LongTensor]
    candidates: Optional[torch.LongTensor]
    item_seq: Optional[torch.LongTensor]
    vae_seq: Optional[torch.LongTensor]
    labels: Optional[torch.LongTensor]
    years: Optional[torch.LongTensor]
    months: Optional[torch.LongTensor]
    days: Optional[torch.LongTensor]
    seasons: Optional[torch.LongTensor]
    hours: Optional[torch.LongTensor]
    minutes: Optional[torch.LongTensor]
    seconds: Optional[torch.LongTensor]
    user_matrix: Optional[torch.FloatTensor]

    class Config:
        arbitrary_types_allowed = True


class TVASequenceDataset(Dataset):
    def __init__(
        self,
        u2seq: Dict[int, List[int]],
        u2timeseq: Dict[int, List[int]],
        max_len: int,
        mask_token: int,
        mode: str = "train",  # train, eval, inference
        # Train parameters
        num_items: Optional[int] = 0,
        mask_prob: Optional[float] = 0.0,
        seed: Optional[int] = 0,
        user_matrix: Optional[np.ndarray] = None,
        # Eval parameters
        u2answer: Optional[Dict[int, List[int]]] = None,
        u2eval_time: Optional[Dict[int, List[int]]] = None,
    ) -> None:

        if mode == "eval":
            if u2answer is None and num_items == 0:
                raise ValueError("num_items, u2answer must be provided")
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
        self.u2answer = u2answer
        self.u2time_seq = u2timeseq
        self.u2eval_time = u2eval_time
        self.user_matrix = user_matrix

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:

        user = self.users[index]
        seq = self.u2seq[user]

        # Time interval sequence
        time_seq = self.u2time_seq[user]

        time_interval_seq = [0] + [
            int((time_seq[i] - time_seq[i - 1]) / 100000)
            for i in range(1, len(time_seq))
        ]

        if self.mode == "train":
            train_item_seq = []
            labels = []

            train_time_interval_seq = []
            train_time_interval_seq = time_interval_seq

            train_time_seq = time_seq

            # Do masking
            for idx, s in enumerate(seq):
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        train_item_seq.append(self.mask_token)

                    elif prob < 0.9:
                        train_item_seq.append(self.rng.randint(1, self.num_items))

                    else:
                        train_item_seq.append(s)

                    labels.append(s)
                else:
                    train_item_seq.append(s)
                    labels.append(0)

            train_item_seq = train_item_seq[-self.max_len :]
            labels = labels[-self.max_len :]

            train_time_seq = train_time_seq[-self.max_len :]
            train_time_interval_seq = train_time_interval_seq[-self.max_len :]

            mask_len = self.max_len - len(train_item_seq)

            train_item_seq = [0] * mask_len + train_item_seq
            labels = [0] * mask_len + labels

            train_time_seq = [0] * mask_len + train_time_seq
            train_time_interval_seq = [0] * mask_len + train_time_interval_seq

            # FIXME: This is a temporary solution
            dates = [datetime.datetime.fromtimestamp(t) for t in train_time_seq]

            years = [d.year if d is not None else 0 for d in dates]
            months = [d.month if d is not None else 0 for d in dates]
            days = [d.day if d is not None else 0 for d in dates]
            hours = [d.hour if d is not None else 0 for d in dates]
            minutes = [d.minute if d is not None else 0 for d in dates]
            seconds = [d.second if d is not None else 0 for d in dates]

            seasons = []
            for month in months:
                if month in [3, 4, 5]:
                    seasons.append(1)
                elif month in [6, 7, 8]:
                    seasons.append(2)
                elif month in [9, 10, 11]:
                    seasons.append(3)
                else:  # month in [12, 1, 2]
                    seasons.append(4)

            user_matrix = torch.FloatTensor(self.user_matrix[user].toarray()[0])

            data = {
                "item_seq": torch.LongTensor(train_item_seq),
                "time_seq": torch.FloatTensor(train_time_seq),
                "time_interval_seq": torch.LongTensor(train_time_interval_seq),
                "labels": torch.LongTensor(labels),
                "user_matrix": user_matrix,
                # Time features
                "years": torch.LongTensor(years),
                "months": torch.LongTensor(months),
                "days": torch.LongTensor(days),
                "hours": torch.LongTensor(hours),
                "minutes": torch.LongTensor(minutes),
                "seconds": torch.LongTensor(seconds),
                "seasons": torch.LongTensor(seasons),
            }

            return TVASequences(**data).dict(exclude_none=True)

        if self.mode == "eval":
            # candidates: candidates from negative sampling
            answer = self.u2answer[user]
            interacted = list(set(seq))
            interacted += answer

            negs = [x for x in range(1, self.num_items + 1) if x not in interacted]

            # if negs is not enough, pad with the last negative item
            if len(negs) < self.num_items:
                negs += [0] * (self.num_items - len(negs))

            candidates = answer + negs

            # labels: labels from user's answer and negative samples
            labels = [1] * len(answer) + [0] * len(negs)

            # seq: user's item sequence
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            # FIXME
            val_time = self.u2eval_time[user]
            time_seq = time_seq + val_time
            time_seq = time_seq[-self.max_len :]
            time_seq = [0] * padding_len + time_seq

            # time_interval_seq: time intervals of user's sequence
            time_interval_seq = time_interval_seq + [0]
            time_interval_seq = time_interval_seq[-self.max_len :]
            time_interval_seq = [0] * padding_len + time_interval_seq

            # FIXME: This is a temporary solution
            dates = [datetime.datetime.fromtimestamp(t) for t in time_seq]
            years = [d.year if d is not None else 0 for d in dates]
            months = [d.month if d is not None else 0 for d in dates]
            days = [d.day if d is not None else 0 for d in dates]
            hours = [d.hour if d is not None else 0 for d in dates]
            minutes = [d.minute if d is not None else 0 for d in dates]
            seconds = [d.second if d is not None else 0 for d in dates]

            seasons = []
            for month in months:
                if month in [3, 4, 5]:
                    seasons.append(1)
                elif month in [6, 7, 8]:
                    seasons.append(2)
                elif month in [9, 10, 11]:
                    seasons.append(3)
                else:  # month in [12, 1, 2]
                    seasons.append(4)

            user_matrix = torch.FloatTensor(self.user_matrix[user].toarray()[0])

            data = {
                "item_seq": torch.LongTensor(seq),
                "time_seq": torch.FloatTensor(time_seq),
                "time_interval_seq": torch.LongTensor(time_interval_seq),
                "candidates": torch.LongTensor(candidates),
                "labels": torch.LongTensor(labels),
                "user_matrix": user_matrix,
                # Time features
                "years": torch.LongTensor(years),
                "months": torch.LongTensor(months),
                "days": torch.LongTensor(days),
                "hours": torch.LongTensor(hours),
                "minutes": torch.LongTensor(minutes),
                "seconds": torch.LongTensor(seconds),
                "seasons": torch.LongTensor(seasons),
            }

            return TVASequences(**data).dict(exclude_none=True)

        if self.mode == "inference":
            # candidates = self.negative_samples[user]
            candidates = [x for x in range(1, self.num_items + 1)]
            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            user_matrix = torch.FloatTensor(self.user_matrix[user].toarray()[0])

            data = {
                "item_seq": torch.LongTensor(seq),
                "time_seq": torch.FloatTensor(train_time_seq),
                "time_interval_seq": torch.LongTensor(train_time_interval_seq),
                "candidates": torch.LongTensor(candidates),
                "user_matrix": user_matrix,
            }

            return TVASequences(**data).dict(exclude_none=True)
