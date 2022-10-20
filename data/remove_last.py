#%%
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("./carrefour_sales.csv")

#%%
df["product"] = df.apply(lambda x: [x["product"]], axis=1)
gdf = (
    df.groupby(["customer", "order_date"], group_keys=False)
    .agg({"product": "sum"})
    .reset_index()
)

gdf["product"]  = gdf.apply(lambda x: ",".join(x["product"]), axis=1)
last = gdf.drop_duplicates(subset="customer", keep="last")
#%%
last
#%%
last.to_csv("last_buy.csv", index=False)
#%%
drop_idxs = []

for idx, row in tqdm(last.iterrows(), total=len(last)):
    # print(row)
    # drop_idx = df[
    #     (df["customer"] == row["customer"]) & (df["order_date"] == row["order_date"])
    # ].index
    udf = df[df["customer"] == row["customer"]]
    drop_idx = udf[udf["order_date"] == row["order_date"]].index

    drop_idxs.extend(drop_idx)
    # df.drop(drop_idx, inplace=True)

# %%
df.drop(drop_idx, inplace=True)
df.to_csv("without_last.csv", index=False)
