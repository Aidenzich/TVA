import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from config import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)


class Data:
    def __init__(self, df: pd.DataFrame):
        self.dataframe = df
        (
            self.u2cat,
            self.i2cat,
            self.cat2u,
            self.cat2i,
            self.dataframe[USER_COLUMN_NAME],
            self.dataframe[ITEM_COLUMN_NAME],
        ) = self._get_cat2id(df)

        self.num_users = len(self.u2cat)
        self.num_items = len(self.i2cat)

    def _get_cat2id(
        self, df: pd.DataFrame
    ) -> tuple[dict, dict, dict, dict, pd.core.series.Series, pd.core.series.Series]:
        users_cats = df[USER_COLUMN_NAME].astype("category").cat.codes
        items_cats = df[ITEM_COLUMN_NAME].astype("category").cat.codes
        u2cat = dict(zip(self.dataframe[USER_COLUMN_NAME], users_cats))
        i2cat = dict(zip(self.dataframe[ITEM_COLUMN_NAME], items_cats))
        cat2u = {v: k for k, v in u2cat.items()}
        cat2i = {v: k for k, v in i2cat.items()}

        return u2cat, i2cat, cat2u, cat2i, users_cats, items_cats


class SequenceDataset(Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._get_user_seq(user)

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

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _get_user_seq(self, user):
        return self.u2seq[user]
