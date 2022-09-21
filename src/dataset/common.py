from typing import Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from ..config import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)


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
        (
            self.train_seqs,
            self.val_seqs,
            self.test_seqs,
            self.users_seqs,
        ) = self._split_df_u2seq(split_method="leave_one_out")

    def _split_df_u2seq(
        self, split_method: str = "leave_one_out"
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Split the dataframe into train, val, and test dictionary of user's trading sequences
        """
        train_seqs, val_seqs, test_seqs, fully_seqs = {}, {}, {}, {}
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
                train_seqs[user], val_seqs[user], test_seqs[user], fully_seqs[user] = (
                    items[:-2],
                    items[-2:-1],
                    items[-1:],
                    items,
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
            train_seqs = dict(
                train_df.groupby(USER_COLUMN_NAME).progress_apply(
                    lambda d: list(d[ITEM_COLUMN_NAME])
                )
            )
            val_seqs = dict(
                val_df.groupby(USER_COLUMN_NAME).progress_apply(
                    lambda d: list(d[ITEM_COLUMN_NAME])
                )
            )
            test_seqs = dict(
                test_df.groupby(USER_COLUMN_NAME).progress_apply(
                    lambda d: list(d[ITEM_COLUMN_NAME])
                )
            )
            fully_seqs = dict(
                self.dataframe.groupby(USER_COLUMN_NAME).progress_apply(
                    lambda d: list(d[ITEM_COLUMN_NAME])
                )
            )

        return train_seqs, val_seqs, test_seqs, fully_seqs

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
