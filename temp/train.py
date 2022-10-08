# %%
import sys

sys.path.append("../")


import cornac
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from src.datasets.common import RecsysData
from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.models.BERT4Rec.model import BERTModel
from src.configs import DATA_PATH, LOG_PATH

#%%
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
# ml_100k = cornac.datasets.movielens.load_feedback(fmt="UIRT")
# np_data = np.array(ml_100k)
# pd_data = pd.DataFrame(
#     {
#         "user_id": np_data[:, 0],
#         "item_id": np_data[:, 1],
#         "rating": np_data[:, 2].astype(float),
#         "timestamp": np_data[:, 3].astype(int),
#     }
# )
# pd_data.info()
# %%
# pd_data.user_id.value_counts().plot(kind="hist")
# pd_data.user_id.value_counts()

# %%
max_len = 512
myData = RecsysData(pd_data)
print(myData.num_items)
print(myData.num_users)
myData.save()
#%%
trainset = SequenceDataset(
    mode="train",
    max_len=max_len,
    mask_prob=0.15,
    num_items=myData.num_items,
    mask_token=myData.num_items + 1,
    u2seq=myData.train_seqs,
    seed=12345,
)

test_negative_sampler = NegativeSampler(
    train=myData.train_seqs,
    val=myData.val_seqs,
    test=myData.test_seqs,
    user_count=myData.num_users,
    item_count=myData.num_items,
    sample_size=128,
    method="random",
    seed=12345,
)
test_negative_samples = test_negative_sampler.get_negative_samples()


valset = SequenceDataset(
    mode="eval",
    mask_token=myData.num_items + 1,
    u2seq=myData.train_seqs,
    u2answer=myData.val_seqs,
    max_len=max_len,
    negative_samples=test_negative_samples,
)

mymodel = BERTModel(
    hidden_size=256,
    num_items=myData.num_items,  # item 的數量
    n_layers=2,
    dropout=0,
    heads=8,
    max_len=max_len,
)


# %%
train_loader = DataLoader(trainset, batch_size=12, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=12, shuffle=False, pin_memory=True)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=LOG_PATH)
# trainer = pl.Trainer(limit_train_batches=100, max_epochs=40, gpus=1, logger=tb_logger)
trainer = pl.Trainer(
    limit_train_batches=100, max_epochs=40, accelerator="gpu", logger=tb_logger
)
trainer.fit(mymodel, train_loader, val_loader)

# %%
