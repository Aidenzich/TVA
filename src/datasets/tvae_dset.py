import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Dict, List, Tuple, Any, Optional
import random
from pydantic import BaseModel, ValidationError
import numpy as np
import datetime
from ..datasets.bert_dset import get_masked_seq


class TVAEDomain(BaseModel):
    time_seq: Optional[torch.FloatTensor]
    time_interval_seq: Optional[torch.LongTensor]
    candidates: Optional[torch.LongTensor]
    item_seq: Optional[torch.LongTensor]
    vae_seq: Optional[torch.LongTensor]
    labels: Optional[torch.LongTensor]
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
        u2val=None,
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
        self.u2val = u2val

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        user = self.users[index]
        item_seq = self.u2seq[user]

        if "." in str(user):
            user = int(user.split(".")[0])

        if self.mode == "train":
            user_matrix = torch.FloatTensor(self.user_matrix[user].toarray()[0])
            data = self._get_train(item_seq=item_seq, user_matrix=user_matrix)
            return TVAEDomain(**data).dict(exclude_none=True)

        if self.mode == "eval":
            val_item = self.u2val[user] if self.u2val is not None else None
            if val_item is not None:
                item_seq = item_seq + val_item

            # candidates: candidates from negative sampling
            answer = self.u2answer[user]

            interacted = list(set(item_seq))
            interacted += answer

            candidates = [
                x for x in range(1, self.num_items + 1) if x not in interacted
            ]

            # if negs is not enough, pad with the last negative item
            if len(candidates) < self.num_items:
                candidates += [0] * (self.num_items - len(candidates))

            candidates = answer + candidates

            # labels: labels from user's answer and negative samples
            labels = [1] * len(answer) + [0] * len(candidates)

            # seq: user's item sequence
            item_seq = item_seq + [self.mask_token]
            item_seq = item_seq[-self.max_len :]
            padding_len = self.max_len - len(item_seq)
            item_seq = [0] * padding_len + item_seq

            user_matrix = torch.FloatTensor(self.user_matrix[user].toarray()[0])

            data = {
                "item_seq": torch.LongTensor(item_seq),
                "candidates": torch.LongTensor(candidates),
                "labels": torch.LongTensor(labels),
                "user_matrix": user_matrix,
            }

            return TVAEDomain(**data).dict(exclude_none=True)

    def _get_train(self, item_seq, user_matrix):
        # Do masking
        masked_item_seq, labels = get_masked_seq(
            item_seq=item_seq,
            max_len=self.max_len,
            mask_prob=self.mask_prob,
            mask_token=self.mask_token,
            num_items=self.num_items,
            rng=self.rng,
        )

        data = {
            "item_seq": torch.LongTensor(masked_item_seq),
            "labels": torch.LongTensor(labels),
            "user_matrix": user_matrix,
        }

        return data
