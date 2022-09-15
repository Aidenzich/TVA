# %%
import cornac
import numpy as np
import pandas as pd
from dset import RecsysData, SequenceDataset
from negative_sampler import NegativeSampler
import random
from model import BERTModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# %%
ml_100k = cornac.datasets.movielens.load_feedback(fmt="UIRT")
np_data = np.array(ml_100k)


# %%
pd_data = pd.DataFrame(
    {
        "user_id": np_data[:, 0],
        "item_id": np_data[:, 1],
        "rating": np_data[:, 2].astype(float),
        "timestamp": np_data[:, 3].astype(int),
    }
)
pd_data.info()

# %%
max_len = 100
myData = RecsysData(pd_data)
myData.num_items
trainset = SequenceDataset(
    # Hyperparameters
    max_len=max_len,
    mask_prob=0.15,
    num_items=myData.num_items,
    mask_token=myData.num_items + 1,
    u2seq=myData.train_seqs,
    rng=random.Random(1234),
)

test_negative_sampler = NegativeSampler(
    train=myData.train_seqs,
    val=myData.val_seqs,
    test=myData.test_seqs,
    user_count=myData.num_users,
    item_count=myData.num_items,
    sample_size=120,
    seed=1234,
)
test_negative_samples = test_negative_sampler.get_negative_samples()


valset = SequenceDataset(
    mask_token=myData.num_items + 1,
    eval=True,
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
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=128, shuffle=False, pin_memory=True)

# %%
trainer = pl.Trainer(limit_train_batches=100, max_epochs=10, gpus=1)
trainer.fit(mymodel, train_loader, val_loader)
