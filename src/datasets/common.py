from typing import Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

tqdm.pandas()

from ..configs import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
    DATACLASS_PATH,
)


class RecsysData:
    def __init__(self, df: pd.DataFrame, filename=""):
        self.filename = filename
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
        self.prepare_matrix()

    def _split_matrix_by_user(self):
        print("Splitting")
        # split to train val and test
        users = list(self.u2cat.values())
        import random

        random.shuffle(users)
        train_num = int(len(users) * 0.8)
        test_num = int(len(users) * 0.1)
        # val_num = len(users) - train_num - test_num
        print(len(users[:train_num]))

        train_users = users[:train_num]
        test_users = users[-test_num:]
        val_users = users[train_num:-test_num]

        train_matrix = self.matrix[train_users, :]
        test_matrix = self.matrix[test_users, :]
        val_matrix = self.matrix[val_users, :]

        return train_matrix, test_matrix, val_matrix

    def prepare_matrix(self):
        uir_df = self.dataframe[
            [USER_COLUMN_NAME, ITEM_COLUMN_NAME, RATING_COLUMN_NAME]
        ]
        uir_vals = uir_df.values
        print(uir_vals)
        u_indices, i_indices, r_values = uir_vals[:, 0], uir_vals[:, 1], uir_vals[:, 2]

        from scipy.sparse import csr_matrix

        self.matrix = csr_matrix(
            (r_values, (u_indices, i_indices)),
            shape=(self.num_users, self.num_items),
        )
        (
            self.train_matrix,
            self.test_matrix,
            self.val_matrix,
        ) = self._split_matrix_by_user()

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

    def save(self):
        savefile_path = self._get_save_path()
        with savefile_path.open("wb") as f:
            pickle.dump(self, f)

    def _get_save_path(self):
        self.filename

        savename = self.filename + "_cls.pkl"
        return DATACLASS_PATH / savename

    @staticmethod
    def reverse_ids(data_cls, pred_result):

        reversed_pred_result = {}
        for user in tqdm(pred_result.keys()):
            reversed_pred_result[data_cls.cat2u[user]] = [
                data_cls.cat2i[item] for item in pred_result[user]
            ]

        return reversed_pred_result
