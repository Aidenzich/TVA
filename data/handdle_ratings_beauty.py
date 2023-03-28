#%%
import pandas as pd

df = pd.read_csv(
    "ratings_Beauty.csv",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
)
df
# %%

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
