import torch
from torch.utils.data import Dataset
from torch import Tensor, LongTensor
from typing import Dict, List, Tuple, Any, Optional
import random
from pydantic import BaseModel, ValidationError
import numpy as np
import datetime
from ..datasets.bert_dset import get_masked_seq


class TVASequences(BaseModel):
    userwise_latent_factor: Optional[torch.FloatTensor]
    itemwise_latent_factor_seq: Optional[torch.FloatTensor]
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
        # Eval parameters
        u2answer: Optional[Dict[int, List[int]]] = None,
        u2answer_time: Optional[Dict[int, List[int]]] = None,
        u2val_time: Optional[Dict[int, List[int]]] = None,
        u2val=None,
        # Latent
        user_latent_factor=None,
        item_latent_factor=None,
        # Sliding window
        seqs_user = None
    ) -> None:

        if mode == "eval":
            if u2answer is None and num_items == 0:
                raise ValueError("num_items, u2answer must be provided")
        if mode == "train":
            if num_items == 0 or mask_prob == 0:
                raise ValueError("num_items, mask_prob must be provided")

        self.mode = mode
        self.u2seq = u2seq
        self.users = list(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = random.Random(seed)
        self.u2answer = u2answer
        self.u2time_seq = u2timeseq
        self.u2val_time = u2val_time
        self.u2answer_time = u2answer_time
        self.u2val = u2val
        self.user_latent_factor = user_latent_factor
        self.item_latent_factor = item_latent_factor
        self.zero_latent_factor = np.zeros_like(
            self.item_latent_factor[0]
        )  # (item_latent_factor_dim)
        self.random_latent_factor = np.random.normal(
            0, 0.1, self.item_latent_factor[0].shape
        )
        self.seqs_user = seqs_user

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        user = self.users[index]
        item_seq = self.u2seq[user]        
        time_seq = self.u2time_seq[user]

        if "." in str(user):
            user = int(user.split(".")[0])

        # User latent factor
        user_latent_factor = self.user_latent_factor[user]

        if self.mode == "train":
            data = self._get_train(
                item_seq=item_seq,
                time_seq=time_seq,
                user_latent_factor=user_latent_factor,
            )
            data = TVASequences(**data).dict(exclude_none=True)
            return data

        if self.mode == "eval":
            data = self._get_eval(
                item_seq=item_seq,
                time_seq=time_seq,
                answer_item=self.u2answer[user],
                answer_time=self.u2answer_time[user],
                user_latent_factor=user_latent_factor,
                val_item=None if self.u2val is None else self.u2val[user],
                val_time=None if self.u2val_time is None else self.u2val_time[user],
            )
            data = TVASequences(**data).dict(exclude_none=True)

            return data

    def _get_train(self, item_seq, time_seq, user_latent_factor):
        # Do masking
        masked_item_seq, labels = get_masked_seq(
            item_seq=item_seq,
            max_len=self.max_len,
            mask_prob=self.mask_prob,
            mask_token=self.mask_token,
            num_items=self.num_items,
            rng=self.rng,
        )

        # Bulid Item's latent factor sequence
        masked_item_latent_seq = []
        for item_id in masked_item_seq:
            if item_id == 0:
                masked_item_latent_seq.append(self.zero_latent_factor)
            elif item_id == self.mask_token:
                masked_item_latent_seq.append(self.random_latent_factor)
            else:
                masked_item_latent_seq.append(self.item_latent_factor[item_id])

        # Pad the time sequence
        time_seq = time_seq[-self.max_len :]
        time_seq = [0] * (self.max_len - len(time_seq)) + time_seq

        data = {
            "item_seq": torch.LongTensor(masked_item_seq),
            "time_seq": torch.FloatTensor(time_seq),
            "userwise_latent_factor": torch.FloatTensor(user_latent_factor),
            "itemwise_latent_factor_seq": torch.FloatTensor(
                np.array(masked_item_latent_seq)
            ),
            "labels": torch.LongTensor(labels),
        }

        time_features = self._get_time_features(time_seq)
        data = {**data, **time_features}

        assert (
            len(data["item_seq"])
            == len(data["time_seq"])
            == len(data["itemwise_latent_factor_seq"])
            == len(data["labels"])
        ), "The length of item_seq, time_seq, itemwise_latent_factor_seq, labels, time_features must be equal"

        return data

    def _get_eval(
        self,
        item_seq,
        time_seq,
        answer_item,
        answer_time,
        user_latent_factor,
        val_item=None,
        val_time=None,
    ):
        if val_item is not None and val_time is not None:
            # print("val_item founded", val_item)
            # print("val_time founded", val_time)
            # In test phase, we add val_item to item_seq,
            # and use the item_seq to predict the answer_item
            item_seq = item_seq + val_item
            time_seq = time_seq + val_time

        # Mask the last item, which need to be predicted
        item_seq = item_seq + [self.mask_token]
        # Truncate the sequence to max_len
        item_seq = item_seq[-self.max_len :]
        # Pad the sequence to max_len
        item_seq = [0] * (self.max_len - len(item_seq)) + item_seq

        time_seq += answer_time
        time_seq = time_seq[-self.max_len :]
        time_seq = [0] * (self.max_len - len(time_seq)) + time_seq

        # Get the user interacted items
        interacted = list(set(answer_item + item_seq))

        # Negative sampling: we use all items which not interacted by user
        candidates = answer_item + [
            x for x in range(1, self.num_items + 1) if x not in interacted
        ]

        # If candidates is not enough, pad with 0
        candidates = candidates + [0] * (self.num_items - len(candidates))

        # Generate labels, the first one is the answer, the rest are negative samples
        labels = [1] * len(answer_item) + [0] * (len(candidates) - 1)

        # item_latent_factor_seq: item latent factors of user's sequence
        item_latent_factor_seq = []
        for item_id in item_seq:
            if item_id == 0:
                item_latent_factor_seq.append(self.zero_latent_factor)
            elif item_id == self.mask_token:
                item_latent_factor_seq.append(self.random_latent_factor)
            else:
                item_latent_factor_seq.append(self.item_latent_factor[item_id])

        # Assert
        assert len(item_seq) == len(
            time_seq
        ), f"item_seq {len(item_seq)} and time_seq {len(time_seq)} should have the same length"

        assert len(item_latent_factor_seq) == len(
            item_seq
        ), f"{len(item_latent_factor_seq)} != {len(item_seq)}"

        assert len(candidates) == len(labels), f"{len(candidates)} != {len(labels)}"

        data = {
            # user's sequence
            "item_seq": torch.LongTensor(item_seq),
            # candidates from negative sampling
            "candidates": torch.LongTensor(candidates),
            # labels from user's answer and negative samples
            "labels": torch.LongTensor(labels),
            # "time_seq": torch.FloatTensor(time_seq),
            "userwise_latent_factor": torch.FloatTensor(user_latent_factor),
            "itemwise_latent_factor_seq": torch.FloatTensor(
                np.array(item_latent_factor_seq)
            ),
        }

        time_features = self._get_time_features(time_seq)

        data = {**data, **time_features}

        return data

    # def _get_predict(self, user_latent_factor):
    # candidates = [x for x in range(1, self.num_items + 1)]
    # item_seq = item_seq + [self.mask_token]
    # item_seq = item_seq[-self.max_len :]

    # padding_len = self.max_len - len(item_seq)

    # item_seq = [0] * padding_len + item_seq

    # time_seq = time_seq[-self.max_len :]
    # time_seq = [0] * padding_len + time_seq

    # # item_latent_factor_seq: item latent factors of user's sequence
    # item_latent_factor_seq = []
    # for item_id in item_seq:
    #     if item_id == 0 or item_id == self.mask_token:
    #         item_latent_factor_seq.append(self.zero_latent_factor)
    #     else:
    #         item_latent_factor_seq.append(self.item_latent_factor[item_id])

    # return {
    #     "item_seq": torch.LongTensor(item_seq),
    #     "candidates": torch.LongTensor(candidates),
    #     "time_seq": torch.FloatTensor(time_seq),
    #     "userwise_latent_factor": torch.FloatTensor(user_latent_factor),
    #     "itemwise_latent_factor_seq": torch.FloatTensor(item_latent_factor_seq),
    # }
    # pass

    def _get_time_features(self, time_seq) -> Dict[str, LongTensor]:
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

        return {
            "years": torch.LongTensor(years),
            "months": torch.LongTensor(months),
            "days": torch.LongTensor(days),
            "hours": torch.LongTensor(hours),
            "minutes": torch.LongTensor(minutes),
            "seconds": torch.LongTensor(seconds),
            "seasons": torch.LongTensor(seasons),
        }