#%%
import pandas as pd
from pathlib import Path

myPath = Path(".").parent
print(myPath.absolute())
df = pd.read_pickle(myPath / "data.pkl")

df.order_date = pd.to_datetime(df.order_date)
df["timestamp"] = df.order_date.astype(int) // 10**9

df["timestamp"]

#%%
df.columns

# %%
df.rename(columns={"customer": "user_id"}, inplace=True)

#%%
df.rename(columns={"product": "item_id"}, inplace=True)

#%%
df

#%%
df.to_pickle(myPath / "carrefour.pkl")

# %%
