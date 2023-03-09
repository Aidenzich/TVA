from pathlib import Path
from typing import Dict, List, Any, Tuple
from tabulate import tabulate
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

from ..configs import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
    DATACLASS_PATH,
    RED_COLOR,
    CYAN_COLOR,
    END_COLOR,
)

tqdm.pandas()

class RecsysData:
    def __init__(self, df: pd.DataFrame, filename="") -> None:
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

        # dataset info
        self.max_length = max(df[USER_COLUMN_NAME].value_counts())
        self.min_length = min(df[USER_COLUMN_NAME].value_counts())
        self.mean_length = np.mean(df[USER_COLUMN_NAME].value_counts())
        self.num_users = len(self.u2cat)
        self.num_items = len(self.i2cat)

        (
            self.train_seqs,
            self.val_seqs,
            self.test_seqs,
            self.users_seqs,
            self.train_timeseqs,
            self.val_timeseqs,
            self.test_timeseqs,
            self.users_timeseqs,
        ) = self._split_df_u2seq(split_method="leave_one_out")

        self.prepare_matrix()

        self.seqs_user_num = len(self.train_seqs)

    def show_info_table(self) -> None:
        """
        Show a table of information about the attributes of an object, including the number of users, 
        items, maximum and minimum lengths of sequences, number of sequences per user, and total 
        number of sessions. This information is printed in an organized format using the "orgtbl" table format.
        """
        print(CYAN_COLOR)
        print()
        print(
            tabulate(
                [
                    [
                        "num_users", 
                        self.num_users,
                    ],
                    [
                        "num_items",
                        self.num_items,
                    ],
                    [
                        "mean_length",
                        self.mean_length    
                    ],
                    [
                        "max_length",
                        self.max_length
                    ], 
                    [
                        "min_length",
                        self.min_length
                    ], 
                    [
                        "seqs_user_num",
                        self.seqs_user_num
                    ], 
                    [
                        "total_session_num",
                        len(self.dataframe)
                    ]
                ],
                headers=[
                    "property", "value"
                ],
                tablefmt="heavy_outline",
                numalign="right"
            ),
            
        )
        print(END_COLOR)

    def _split_matrix_by_user(self) -> Tuple[Any, Any, Any]:
        """
        Splitting matrix by random shuffle user for train, testing and validation
        """

        print(
            "Splitting matrix by random shuffle user for train, testing and validation"
        )

        # split to train val and test
        users = list(self.u2cat.values())
        import random

        random.shuffle(users)
        train_num = int(len(users) * 0.98)
        test_num = int(len(users) * 0.01)
        # val_num = len(users) - train_num - test_num
        # print(len(users[:train_num]))

        train_users = users[:train_num]
        test_users = users[-test_num:]
        val_users = users[train_num:-test_num]

        train_matrix = self.matrix[train_users, :]
        test_matrix = self.matrix[test_users, :]
        val_matrix = self.matrix[val_users, :]

        return train_matrix, test_matrix, val_matrix

    def _filter_rating_threshold(self, df: pd.DataFrame, threshold=5) -> pd.DataFrame:
        return df[df[RATING_COLUMN_NAME] >= threshold]

    def _filter_purchases_threshold(self, df: pd.DataFrame, threshold=10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        user_val_count = df[USER_COLUMN_NAME].value_counts()
        user_index = user_val_count[user_val_count > threshold].index
        remove_user_index = user_val_count[user_val_count <= threshold].index
        removed_df = df[df[USER_COLUMN_NAME].isin(remove_user_index)]
        filtered_df = df[df[USER_COLUMN_NAME].isin(user_index)]

        return filtered_df, removed_df

    def prepare_matrix(self) -> None:
        uir_df = self.dataframe[
            [USER_COLUMN_NAME, ITEM_COLUMN_NAME, RATING_COLUMN_NAME]
        ]

        uir_df[RATING_COLUMN_NAME] = uir_df[RATING_COLUMN_NAME].astype(float)
        uir_df = uir_df[uir_df[RATING_COLUMN_NAME] > 0]

        drop_all_idx = []

        # Remove the last 2 items from each user's sequence for sequence test and validation
        for u in tqdm(self.val_seqs):
            # before = len(uir_df[uir_df[USER_COLUMN_NAME] == u])
            udf = uir_df[uir_df[USER_COLUMN_NAME] == u]
            val = udf.values
            drop_idx = np.where(
                (
                    (val[:, 1] == self.val_seqs[u][0])
                    | (val[:, 1] == self.test_seqs[u][0])
                )
            )

            drop_all_idx.extend(list(drop_idx[0]))

            # after = len(uir_df[uir_df[USER_COLUMN_NAME] == u])
            # print(before, after)

        uir_vals = uir_df.values
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
    ) -> Tuple[Dict[int, List[int]]]:
        """
        Split the dataframe into train, val, test and total dictionary of user's trading sequences
        """
        print("Splitting user sequences...")
        df, self.removed_df = self._filter_purchases_threshold(
            self.dataframe, threshold=3
        )
        train_seqs, val_seqs, test_seqs, fully_seqs = {}, {}, {}, {}
        train_timeseqs, val_timeseqs, test_timeseqs, fully_timeseqs = {}, {}, {}, {}
        users = df[USER_COLUMN_NAME].unique()

        if split_method == "leave_one_out":

            user_group = df.groupby(USER_COLUMN_NAME)
            user2items = user_group.progress_apply(
                lambda d: list(
                    d.sort_values(by=TIMESTAMP_COLUMN_NAME)[ITEM_COLUMN_NAME]
                )
            )

            user2time = user_group.progress_apply(
                lambda t: list(
                    t.sort_values(by=TIMESTAMP_COLUMN_NAME)[
                        TIMESTAMP_COLUMN_NAME
                    ].astype(np.int64)
                )
            )

            for user in users:
                items = user2items[user]
                timestamps = user2time[user]

                train_seqs[user], val_seqs[user], test_seqs[user], fully_seqs[user] = (
                    items[:-2],
                    items[-2:-1],
                    items[-1:],
                    items,
                )

                (
                    train_timeseqs[user],
                    val_timeseqs[user],
                    test_timeseqs[user],
                    fully_timeseqs[user],
                ) = (
                    timestamps[:-2],
                    timestamps[-2:-1],
                    timestamps[-1:],
                    timestamps,
                )

        # elif split_method == "holdout":
        #     print("Splitting")
        #     np.random.seed(self.args.dataset_split_seed)
        #     eval_set_size = self.args.eval_set_size

        #     # Generate user indices
        #     permuted_index = np.random.permutation(user_count)
        #     train_user_index = permuted_index[: -2 * eval_set_size]
        #     val_user_index = permuted_index[-2 * eval_set_size : -eval_set_size]
        #     test_user_index = permuted_index[-eval_set_size:]

        #     # Split DataFrames
        #     train_df = df.loc[df[USER_COLUMN_NAME].isin(train_user_index)]

        #     val_df = df.loc[df[USER_COLUMN_NAME].isin(val_user_index)]

        #     test_df = df.loc[df[USER_COLUMN_NAME].isin(test_user_index)]

        #     # DataFrame to dict
        #     train_seqs = dict(
        #         train_df.groupby(USER_COLUMN_NAME).progress_apply(
        #             lambda d: list(d[ITEM_COLUMN_NAME])
        #         )
        #     )
        #     val_seqs = dict(
        #         val_df.groupby(USER_COLUMN_NAME).progress_apply(
        #             lambda d: list(d[ITEM_COLUMN_NAME])
        #         )
        #     )
        #     test_seqs = dict(
        #         test_df.groupby(USER_COLUMN_NAME).progress_apply(
        #             lambda d: list(d[ITEM_COLUMN_NAME])
        #         )
        #     )
        #     fully_seqs = dict(
        #         self.dataframe.groupby(USER_COLUMN_NAME).progress_apply(
        #             lambda d: list(d[ITEM_COLUMN_NAME])
        #         )
        #     )

        return (
            train_seqs,
            val_seqs,
            test_seqs,
            fully_seqs,
            train_timeseqs,
            val_timeseqs,
            test_timeseqs,
            fully_timeseqs,
        )

    def _get_cat2id(
        self, df: pd.DataFrame
    ) -> Tuple[
            Dict[str, int], 
            Dict[str, int], 
            Dict[int, str], 
            Dict[int,str], 
            pd.core.series.Series, 
            pd.core.series.Series
        ]:
        """
        The function of the following code is to convert the contents of the USER_COLUMN_NAME and
        ITEM_COLUMN_NAME columns in the passed DataFrame to numerical category codes using 
        `astype("category").cat.codes`, and create the relevant conversion dictionaries. 
        The converted results are then returned.
        """
        users_cats = df[USER_COLUMN_NAME].astype("category").cat.codes
        items_cats = df[ITEM_COLUMN_NAME].astype("category").cat.codes
        u2cat = dict(zip(self.dataframe[USER_COLUMN_NAME].astype("str"), users_cats))
        i2cat = dict(zip(self.dataframe[ITEM_COLUMN_NAME].astype("str"), items_cats))
        cat2u = {v: k for k, v in u2cat.items()}
        cat2i = {v: k for k, v in i2cat.items()}

        return u2cat, i2cat, cat2u, cat2i, users_cats, items_cats

    def save(self) -> None:
        savefile_path = self._get_save_path()
        with savefile_path.open("wb") as f:
            pickle.dump(self, f)

    def _get_save_path(self) -> Path:
        savename = self.filename + "_cls.pkl"
        return DATACLASS_PATH / savename

    @staticmethod
    def reverse_ids(data_cls, pred_result) -> Dict[int, List[int]]:
        """
        # For each user, map their category to the corresponding ID using the data_cls.cat2u mapping
        # and store the result as the key in the reversed_pred_result dictionary
        # then map the item's category to the corresponding ID using the data_cls.cat2i mapping
        """
        reversed_pred_result = {}
                
        for user in tqdm(pred_result.keys()):
            reversed_pred_result[data_cls.cat2u[user]] = [
                data_cls.cat2i[item] for item in pred_result[user]
            ]

        return reversed_pred_result

    @staticmethod
    def convert_ids(data_cls, samples) -> Dict[int, List[int]]:
        """
        # Iterate over the keys (users) in the input samples dictionary
        # For each user, map their ID to the corresponding category using the data_cls.u2cat mapping
        # and store the result as the key in the converted_samples dictionary
        # then map the item's ID to the corresponding category using the data_cls.i2cat mapping
        # and store the result in a list
        """
        converted_samples = {}
                
        for user in tqdm(samples.keys()):            
            converted_samples[data_cls.u2cat[user]] = [
                data_cls.i2cat[item] for item in samples[user]
            ]

        return converted_samples
    