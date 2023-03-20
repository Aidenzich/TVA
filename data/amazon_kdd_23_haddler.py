#%%
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("./sessions_train.csv", header=None)


df
# %%
uir_df = {
    "user": [],
    "item": [],
    "rating": [],
    "time": [],
}


for u in tqdm(df.index):
    if u == 0:  # Skip header
        continue

    udata = df.iloc[u]
    history = (
        udata[0]
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace(" ", ",")
        .split(",")
    )

    for i in history:
        uir_df["user"].append(u)
        uir_df["item"].append(i)
        uir_df["rating"].append(1)
        uir_df["time"].append(0)

    uir_df["user"].append(u)
    uir_df["item"].append(udata[1])
    uir_df["rating"].append(1)
    uir_df["time"].append(0)


#%%
pd.DataFrame(uir_df).to_csv("./amazon23_uir_train.csv", index=False)
