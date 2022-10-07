# %%
import sys

sys.path.append("../")
sys.path.append("../../")


import cornac
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from src.datasets.common import RecsysData
from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.models.BERT4Rec.model import BERTModel
from src.config import DATA_PATH, LOG_PATH

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
