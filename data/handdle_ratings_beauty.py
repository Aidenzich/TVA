#%%
import pandas as pd

pd.read_csv(
    "ratings_Beauty.csv",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"],
).to_csv("ratings_Beauty2.csv", index=False)

# %%
