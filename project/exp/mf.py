# %%
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm
import sys
import pickle
from utils import recall_at_k, ndcg_at_k

sys.path.append("../../")
from src.configs import DATACLASS_PATH

dataset = "toys.pkl"


class MatrixFactorization:
    """Matrix Factorization
        from: Matrix Factorization Techniques for Recommender Systems

    Parameters
    ----------
    train_df: pd.DataFrame, Training DataFrame, with columns =
        [pivot_index_name, pivot_columns_name, pivot_values_name]
    svds_k: int, Hyperparameter K for svds
    pivot_index_name: str, user columns name
    pivot_columns_name: str, item columns name
    pivot_values_name: str, rank columns name
    """

    def __init__(
        self,
        train_df,
        svds_k,
        pivot_index_name="user_id",
        pivot_columns_name="item_id",
        pivot_values_name="rating",
    ) -> None:
        self.name = "MatrixFactorization"
        self.result_df = pd.DataFrame()
        self.svds_k = svds_k
        self.pivot_index = pivot_index_name
        self.pivot_columns = pivot_columns_name
        self.pivot_values = pivot_values_name
        self.fit(train_df)

    def fit(self, train_df) -> None:
        u_i_p_matrix_df = train_df.pivot(
            index=self.pivot_index, columns=self.pivot_columns, values=self.pivot_values
        ).fillna(0)

        u_i_p_matrix = u_i_p_matrix_df.values
        u_i_p_sparse_matrix = csr_matrix(u_i_p_matrix)

        # do svd
        U, sigma, Vt = svds(u_i_p_sparse_matrix, k=self.svds_k)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        # normalize
        all_user_predicted_ratings_norm = (
            all_user_predicted_ratings - all_user_predicted_ratings.min()
        ) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

        # result
        self.result_df = pd.DataFrame(
            all_user_predicted_ratings_norm,
            columns=u_i_p_matrix_df.columns,
            index=list(u_i_p_matrix_df.index),
        ).transpose()

    def rec_items(self, user, items_to_ignore=[], topn=10) -> pd.DataFrame:
        """recommender items by user's id
        Parameters
        ----------
        user: str, user's id.
        items_to_ignore: list, list of items which you want to ignore.
        topn: int, The number of items you want to recommend.
        """
        sorted_user_preds = (
            self.result_df[user]
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={user: self.pivot_values})
        )
        # print(sorted_user_preds)
        ignore = sorted_user_preds[self.pivot_columns].isin(items_to_ignore)
        rec_df = (
            sorted_user_preds[~ignore]
            .sort_values(self.pivot_values, ascending=False)
            .head(topn)
        )

        return rec_df


# %%
with open(DATACLASS_PATH / dataset, "rb") as f:
    dataset = pickle.load(f)

df = dataset.dataframe
for user in tqdm(dataset.test_seqs):
    # popout the test seq:
    for item in dataset.test_seqs[user]:
        df = df.drop(df[(df.user_id == user) & (df.item_id == item)].index)

# dataset.test_seqs is a dictionary where each value is a list
test_seqs_df = pd.DataFrame(
    [(k, i) for k, v in dataset.test_seqs.items() for i in v],
    columns=["user_id", "item_id"],
)
df = df.merge(test_seqs_df, on=["user_id", "item_id"], how="left", indicator=True)
df = df[df["_merge"] == "left_only"]
df.drop(columns="_merge", inplace=True)

print("dropped dataframe len", len(df))

pop = (df.item_id.value_counts() / len(df)).tolist()


# %%
use_seen = False

for svdk in [5, 10, 50, 100, 200, 500]:
    print(f"========== k = {svdk} ==========")
    sum_recall_5, sum_recall_10, sum_recall_20 = 0, 0, 0
    sum_ndcg_5, sum_ndcg_10, sum_ndcg_20 = 0, 0, 0
    mf = MatrixFactorization(df, pivot_values_name="timestamp", svds_k=svdk)

    for user in tqdm(dataset.test_seqs):
        if use_seen:
            bought_item = []
        else:
            bought_item = df[df.user_id == user].item_id.unique().tolist()

        pred_list = mf.rec_items(
            user, topn=20, items_to_ignore=bought_item
        ).item_id.tolist()
        true_list = dataset.test_seqs[user]

        recall_5 = recall_at_k(true_list=true_list, pred_list=pred_list, k=5)
        recall_10 = recall_at_k(true_list=true_list, pred_list=pred_list, k=10)
        recall_20 = recall_at_k(true_list=true_list, pred_list=pred_list, k=20)

        ndcg_5 = ndcg_at_k(true_list=true_list, pred_list=pred_list, k=5)
        ndcg_10 = ndcg_at_k(true_list=true_list, pred_list=pred_list, k=10)
        ndcg_20 = ndcg_at_k(true_list=true_list, pred_list=pred_list, k=20)

        sum_recall_5 += recall_5
        sum_recall_10 += recall_10
        sum_recall_20 += recall_20
        sum_ndcg_5 += ndcg_5
        sum_ndcg_10 += ndcg_10
        sum_ndcg_20 += ndcg_20

    ndcg_5 = sum_ndcg_5 / len(dataset.test_seqs)
    ndcg_10 = sum_ndcg_10 / len(dataset.test_seqs)
    ndcg_20 = sum_ndcg_20 / len(dataset.test_seqs)
    recall_5 = sum_recall_5 / len(dataset.test_seqs)
    recall_10 = sum_recall_10 / len(dataset.test_seqs)
    recall_20 = sum_recall_20 / len(dataset.test_seqs)
    print("ndcg@5", ndcg_5)
    print("ndcg@10", ndcg_10)
    print("ndcg@20", ndcg_20)
    print("recall@5", recall_5)
    print("recall@10", recall_10)
    print("recall@20", recall_20)
