# %%
import sys

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")


import cornac
import numpy as np
import pandas as pd
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.dataset.common import RecsysData
from src.dataset.seq_dset import SequenceDataset
from src.model.BERT4Rec.negative_sampler import NegativeSampler
from src.model.BERT4Rec.model import BERTModel
from src.config import DATA_PATH

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
max_len = 128
myData = RecsysData(pd_data)
##### INFER ######
from tqdm import tqdm
import torch

torch.cuda.empty_cache()

test_negative_sampler = NegativeSampler(
    train=myData.train_seqs,
    val=myData.val_seqs,
    test=myData.test_seqs,
    user_count=myData.num_users,
    item_count=myData.num_items,
    sample_size=120,
    seed=12345,
)
test_negative_samples = test_negative_sampler.get_negative_samples()


device = torch.device("cuda:0")
inferset = SequenceDataset(
    mode="inference",
    mask_token=myData.num_items + 1,
    u2seq=myData.train_seqs,
    u2answer=myData.val_seqs,
    max_len=max_len,
    negative_samples=test_negative_samples,
)

infer_loader = DataLoader(inferset, batch_size=100, shuffle=False, pin_memory=True)

mymodel = BERTModel.load_from_checkpoint(
    "lightning_logs/version_0/checkpoints/epoch=9-step=1000.ckpt"
)


mymodel.to(device)
predict_result: dict = {}
ks = 10

for batch in tqdm(infer_loader):
    seqs, candidates, users = batch
    seqs, candidates, users = seqs.to(device), candidates.to(device), users.to(device)
    scores = mymodel(seqs)
    scores = scores[:, -1, :]  # B x V
    scores = scores.gather(1, candidates)  # B x C
    rank = (-scores).argsort(dim=1)
    predict = candidates.gather(1, rank)
    predict_dict = {
        u: predict[idx].tolist()[:ks]
        for idx, u in enumerate(users.cpu().numpy().flatten())
    }
    predict_result.update(predict_dict)
# %%
