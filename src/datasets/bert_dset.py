import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import random


class BertDataset(Dataset):
    def __init__(
        self,
        u2seq,
        max_len: int,
        mask_token,
        mode="train",  # train, eval, predict
        num_items=0,
        # Train
        mask_prob=0,
        seed=0,
        # Evaluate
        u2answer=None,
        u2test=None,
        u2val=None,
    ) -> None:

        if mode == "eval":
            if u2answer is None:
                raise ValueError("u2answer must be provided")
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
        self.u2test = u2test
        self.u2val = u2val

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[index]
        item_seq = self.u2seq[user]

        if self.mode == "eval":
            answer_item = self.u2answer[user]

            return self._eval(
                item_seq=item_seq,
                answer_item=answer_item,
                val_item=None if self.u2val is None else self.u2val[user],
            )

        if self.mode == "train":
            return self._train(item_seq=item_seq)

        if self.mode == "predict":
            return self._predict(item_seq=item_seq)

    def _train(self, item_seq):
        masked_item_seq = []
        labels = []

        # Mask the item in the sequence
        for item in item_seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    masked_item_seq.append(self.mask_token)
                elif prob < 0.9:
                    masked_item_seq.append(self.rng.randint(1, self.num_items))
                else:
                    masked_item_seq.append(item)

                labels.append(item)
            else:
                masked_item_seq.append(item)
                labels.append(0)

        # Truncate the sequence to max_len
        masked_item_seq = masked_item_seq[-self.max_len :]
        labels = labels[-self.max_len :]

        # Pad the sequence to max_len
        mask_len = self.max_len - len(masked_item_seq)
        masked_item_seq = [0] * mask_len + masked_item_seq
        labels = [0] * mask_len + labels

        return {
            "item_seq": torch.LongTensor(masked_item_seq),
            "labels": torch.LongTensor(labels),
        }

    def _eval(
        self, item_seq, answer_item, val_item=None
    ) -> Dict[str, torch.LongTensor]:

        if val_item is not None:
            # In test phase, we add val_item to item_seq,
            # and use the item_seq to predict the answer_item
            item_seq = item_seq + val_item + [self.mask_token]

        # Mask the last item, which need to be predicted
        item_seq = item_seq + [self.mask_token]
        # Truncate the sequence to max_len
        item_seq = item_seq[-self.max_len :]
        # Pad the sequence to max_len
        item_seq = [0] * (self.max_len - len(item_seq)) + item_seq

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

        return {
            # user's sequence
            "item_seq": torch.LongTensor(item_seq),
            # candidates from negative sampling
            "candidates": torch.LongTensor(candidates),
            # labels from user's answer and negative samples
            "labels": torch.LongTensor(labels),
        }

    def _predict(self):
        candidates = [x for x in range(1, self.num_items + 1)]
        item_seq = item_seq + [self.mask_token]
        item_seq = item_seq[-self.max_len :]
        padding_len = self.max_len - len(item_seq)
        item_seq = [0] * padding_len + item_seq

        return {
            "item_seq": torch.LongTensor(item_seq),
            "candidates": torch.LongTensor(candidates),
        }
