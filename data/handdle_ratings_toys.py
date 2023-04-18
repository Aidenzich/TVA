#%%
import pandas as pd

df = pd.read_csv(
    "ratings_Toys_and_Games.csv",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
)
df
#%%
def filter_triplets(df, min_uc=5, min_sc=5):
    print("Filtering triplets")

    user_count = 0
    item_count = 0
    while True:
        if user_count == len(set(df["user_id"])) and item_count == len(
            set(df["item_id"])
        ):
            break
        else:
            user_count = len(set(df["user_id"]))
            item_count = len(set(df["item_id"]))
            print(user_count, item_count)
        if min_sc > 0:

            item_sizes = df.groupby("item_id").size()
            good_items = item_sizes.index[item_sizes >= min_sc]
            df = df[df["item_id"].isin(good_items)]

        if min_uc > 0:
            user_sizes = df.groupby("user_id").size()
            good_users = user_sizes.index[user_sizes >= min_uc]
            df = df[df["user_id"].isin(good_users)]

    return df


df = filter_triplets(df)

# %%
df.to_csv("toys.csv", index=False)
# Remove the users with less than 5 ratings
df = df.groupby("user_id").filter(lambda x: len(x) >= 5)
df
#%%

# Remove the items with less than 5 ratings
df = df.groupby("item_id").filter(lambda x: len(x) >= 5)
df

#%%
print(df.user_id.nunique())
print(df.item_id.nunique())


#%%
df.to_csv("ratings_Beauty_5.csv", index=False)
