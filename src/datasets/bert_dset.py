import torch
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
        num_mask=1,
    ) -> None:

        if mode == "eval":
            if u2answer is None:
                raise ValueError("u2answer must be provided")
        if mode == "train":
            if num_items == 0 or mask_prob == 0:
                raise ValueError("num_items, mask_prob must be provided")

        self.mode = mode
        self.u2seq = u2seq
        self.users = sorted(list(self.u2seq.keys()))

        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = random.Random(seed)
        self.u2answer = u2answer
        self.u2test = u2test
        self.u2val = u2val

        self.num_mask = num_mask

    def __len__(self) -> int:
        return len(self.u2seq)

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
            return self._get_train(item_seq=item_seq)

        if self.mode == "predict":
            return self._predict(item_seq=item_seq)

    def _get_train(self, item_seq):

        return_dict = {}
        for idx in range(self.num_mask):
            # Do masking
            masked_item_seq, labels = get_masked_seq(
                item_seq=item_seq,
                max_len=self.max_len,
                mask_prob=self.mask_prob,
                mask_token=self.mask_token,
                num_items=self.num_items,
                rng=self.rng,
            )

            return_dict[f"item_seq_{idx}"] = torch.LongTensor(masked_item_seq)
            return_dict[f"labels_{idx}"] = torch.LongTensor(labels)

        return return_dict
        # return {
        #     "item_seq": torch.LongTensor(masked_item_seq),
        #     "labels": torch.LongTensor(labels),
        # }

    def _eval(
        self, item_seq, answer_item, val_item=None
    ) -> Dict[str, torch.LongTensor]:

        if val_item is not None:
            # print("val_item", val_item)
            # In test phase, we add val_item to item_seq,
            # and use the item_seq to predict the answer_item
            item_seq = item_seq + val_item

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


def get_masked_seq(
    item_seq: list,
    rng: random.Random,
    mask_prob: float,
    mask_token: int,
    num_items: int,
    max_len: int,
) -> Tuple[List[int], List[int]]:
    """
    Mask the item in the sequence
    """
    masked_item_seq = []
    labels = []

    for s in item_seq:
        prob = rng.random()
        if prob < mask_prob:
            prob /= mask_prob

            if prob < 0.8:
                masked_item_seq.append(mask_token)

            elif prob < 0.9:
                masked_item_seq.append(rng.randint(1, num_items))

            else:
                masked_item_seq.append(s)

            labels.append(s)
        else:
            masked_item_seq.append(s)
            labels.append(0)

        # Truncate the sequence to max_len
        masked_item_seq = masked_item_seq[-max_len:]
        labels = labels[-max_len:]

        # Pad the sequence to max_len
        mask_len = max_len - len(masked_item_seq)
        masked_item_seq = [0] * mask_len + masked_item_seq
        labels = [0] * mask_len + labels

    return masked_item_seq, labels
