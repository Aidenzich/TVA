#%%
import string
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
from torch.utils.data import Dataset, DataLoader
import torch
from config import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)
from typing import Dict, List, Tuple

#%%
class RecsysData:
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
        self.train_seqs, self.val_seqs, self.test_seqs = self._split_df_u2seq(
            split_method="leave_one_out"
        )

    def _split_df_u2seq(
        self, split_method: string = "leave_one_out"
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Split the dataframe into train, val, and test dictionary of user's trading sequences
        """
        train, val, test = {}, {}, {}
        user_count = self.dataframe[USER_COLUMN_NAME].nunique()

        if split_method == "leave_one_out":
            print("Splitting")
            user_group = self.dataframe.groupby(USER_COLUMN_NAME)
            user2items = user_group.progress_apply(
                lambda d: list(
                    d.sort_values(by=TIMESTAMP_COLUMN_NAME)[ITEM_COLUMN_NAME]
                )
            )

            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = (
                    items[:-2],
                    items[-2:-1],
                    items[-1:],
                )

        elif split_method == "holdout":
            print("Splitting")
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[: -2 * eval_set_size]
            val_user_index = permuted_index[-2 * eval_set_size : -eval_set_size]
            test_user_index = permuted_index[-eval_set_size:]

            # Split DataFrames
            train_df = self.dataframe.loc[
                self.dataframe[USER_COLUMN_NAME].isin(train_user_index)
            ]

            val_df = self.dataframe.loc[
                self.dataframe[USER_COLUMN_NAME].isin(val_user_index)
            ]

            test_df = self.dataframe.loc[
                self.dataframe[USER_COLUMN_NAME].isin(test_user_index)
            ]

            # DataFrame to dict
            train = dict(
                train_df.groupby(USER_COLUMN_NAME).progress_apply(
                    lambda d: list(d[ITEM_COLUMN_NAME])
                )
            )
            val = dict(
                val_df.groupby(USER_COLUMN_NAME).progress_apply(
                    lambda d: list(d[ITEM_COLUMN_NAME])
                )
            )
            test = dict(
                test_df.groupby(USER_COLUMN_NAME).progress_apply(
                    lambda d: list(d[ITEM_COLUMN_NAME])
                )
            )

        return train, val, test

    def _get_cat2id(
        self, df: pd.DataFrame
    ) -> Tuple[Dict, Dict, Dict, Dict, pd.core.series.Series, pd.core.series.Series]:
        users_cats = df[USER_COLUMN_NAME].astype("category").cat.codes
        items_cats = df[ITEM_COLUMN_NAME].astype("category").cat.codes
        u2cat = dict(zip(self.dataframe[USER_COLUMN_NAME], users_cats))
        i2cat = dict(zip(self.dataframe[ITEM_COLUMN_NAME], items_cats))
        cat2u = {v: k for k, v in u2cat.items()}
        cat2i = {v: k for k, v in i2cat.items()}

        return u2cat, i2cat, cat2u, cat2i, users_cats, items_cats


#%%


class SequenceDataset(Dataset):
    def __init__(
        self,
        u2seq,
        max_len,
        mask_token,
        eval=False,
        # for train
        num_items=0,
        mask_prob=0,
        rng=None,
        # for eval
        negative_samples=None,
        u2answer=None,
    ):

        if eval:
            if negative_samples is None or u2answer is None:
                raise ValueError("negative_samples and u2answer must be provided")
        if not eval:
            if num_items == 0 or mask_prob == 0 or rng is None:
                raise ValueError("num_items, mask_prob and rng must be provided")
        self.eval = eval
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.negative_samples = negative_samples
        self.u2answer = u2answer

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[index]
        seq = self.u2seq[user]

        if self.eval:
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

        else:
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


# %%
