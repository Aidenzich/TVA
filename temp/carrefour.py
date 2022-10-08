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

# %%
myData.cat2i
# %%
myData.save()
# %%
