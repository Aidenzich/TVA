# %%
import sys

sys.path.append("../")
sys.path.append("../../")


import pandas as pd
from src.datasets.common import RecsysData
from src.configs import DATA_PATH

# %%
pd_data = pd.read_pickle(DATA_PATH / "carrefour.pkl")
pd_data.rename(columns={"quantity": "rating"}, inplace=True)
pd_data = pd_data[["user_id", "item_id", "rating", "timestamp"]]
pd_data.dropna(inplace=True)

# 為了確保模型不會出錯誤，所以只取 user 交易次數 > 10(min available num) 的資料，這邊應該可以移動到 RecsysData 裡面
vc = pd_data.user_id.value_counts()
user10index = vc[vc > 10].index
print(user10index)


pd_data = pd_data[pd_data.user_id.isin(user10index)]
pd_data
# %%
myData = RecsysData(pd_data)
myData.save()
# %%
print(myData.matrix.shape)
print(myData.train_matrix.shape)
print(myData.test_matrix.shape)
print(myData.val_matrix.shape)

#%%
data = pd.read_csv(DATA_PATH / "carrefour_sales.csv", header=None)

data.columns = [
    "id",
    "order_date",
    "product",
    "sales_price",
    "quantity",
    "department",
    "store",
    "city",
    "district",
    "customer",
    "sex",
    "age_group",
]

#%%
data

# %%
data.to_pickle(DATA_PATH / "carrefour_all.pkl")
data.to_csv(DATA_PATH / "carrefour_sales.csv", index=False)

# %%
# data.order_date.astype("datetime64").astype("int64") // 1000000000
# %%

# str(pd_data.dtypes["timestamp"])
# %%
