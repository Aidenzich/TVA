#%%
import numpy as np
import pandas as pd
import gzip
import tqdm

# This code is from https://github.com/aHuiWang/CIKM2020-S3Rec/blob/master/data/data_process.py
def parse(path):  # for Amazon
    g = gzip.open(path, "r")
    for l in g:
        yield eval(l)


# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    """
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    """
    datas = []
    # older Amazon
    data_flie = dataset_name + ".json.gz"

    # latest Amazon
    # data_flie = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    for inter in parse(data_flie):
        if float(inter["overall"]) <= rating_score:  # 小于一定分数去掉
            continue
        user = inter["reviewerID"]
        item = inter["asin"]
        time = inter["unixReviewTime"]
        datas.append((user, item, int(time)))
    return datas


def get_interaction(datas):
    user_seq = {}
    time_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        times = []
        for t in item_time:
            items.append(t[0])
            times.append(t[1])

        user_seq[user] = items
        time_seq[user] = times

    return user_seq, time_seq


USER_COLUMN_NAME = "user_id"
ITEM_COLUMN_NAME = "item_id"
TIMESTAMP_COLUMN_NAME = "timestamp"
import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def split_df_u2seq(df):
    """
    Split the dataframe into train, val, test and total dictionary of user's trading sequences
    """
    print("Splitting user sequences...")

    train_seqs, val_seqs, test_seqs, fully_seqs = {}, {}, {}, {}
    train_timeseqs, val_timeseqs, test_timeseqs, fully_timeseqs = {}, {}, {}, {}
    users = df[USER_COLUMN_NAME].unique()

    user_group = df.groupby(USER_COLUMN_NAME)

    user2items = user_group.progress_apply(
        lambda d: list(
            d.sort_values(by=TIMESTAMP_COLUMN_NAME, kind="stable")[ITEM_COLUMN_NAME]
        ),
    )

    user2time = user_group.progress_apply(
        lambda t: list(
            t.sort_values(by=TIMESTAMP_COLUMN_NAME, kind="stable")[
                TIMESTAMP_COLUMN_NAME
            ].astype(np.int64)
        )
    )

    # user2items = user2items.to_dict()
    # user2time = user2time.to_dict()

    # for d in user2time:
    #     compare = 0
    #     for t in d:
    #         if t < compare:

    #             assert False, "Timestamps are not sorted"
    #         compare = t

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

    return (
        fully_seqs,
        fully_timeseqs,
    )


#%%
df = pd.read_csv("./beauty.csv")

#%%
fully_seqs, fully_timeseqs = split_df_u2seq(df)
fully_seqs
# %%


#%%
datas = Amazon("reviews_Beauty_5", 0)

#%%
user_seq, time_seq = get_interaction(datas)
user_seq

#%%
print(len(user_seq), len(fully_seqs))

# %%
for u in user_seq:
    if (user_seq[u] == fully_seqs[u]) != True:
        print(user_seq[u])
        print(time_seq[u])
        print(fully_seqs[u])
        print(fully_timeseqs[u])
        print("==================")

# %%
testing = [(1, 1, 2), (1, 2, 2)]
testing.sort(key=lambda x: x[1])
print(testing)
