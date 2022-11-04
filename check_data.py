#%%
import json
import pickle
from src.configs import DATACLASS_PATH

data_class_name = "train_cls.pkl"
print(DATACLASS_PATH)
with open(DATACLASS_PATH / data_class_name, "rb") as f:
    recsys_data = pickle.load(f)
# %%
recsys_data.max_length
# %%
recsys_data.dataframe.user_id.value_counts()
#%%
recsys_data.dataframe[recsys_data.dataframe.user_id == 150567].drop_duplicates
