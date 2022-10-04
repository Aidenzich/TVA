# %%
import sys

sys.path.append("../")
import random
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from src.dataset.common import RecsysData
from src.dataset.seq_dset import SequenceDataset
from src.model.BERT4Rec.negative_sampler import NegativeSampler
from src.model.BERT4Rec.model import BERTModel
from src.config import DATA_PATH, LOG_PATH
import pickle

with open(DATA_PATH / "data_cls.pkl", "rb") as f:
    recsys_data = pickle.load(f)


max_len = 512
#%%
trainset = SequenceDataset(
    mode="train",
    max_len=max_len,
    mask_prob=0.15,
    num_items=recsys_data.num_items,
    mask_token=recsys_data.num_items + 1,
    u2seq=recsys_data.train_seqs,
    rng=random.Random(12345),
)

test_negative_sampler = NegativeSampler(
    train=recsys_data.train_seqs,
    val=recsys_data.val_seqs,
    test=recsys_data.test_seqs,
    user_count=recsys_data.num_users,
    item_count=recsys_data.num_items,
    sample_size=5000,
    method="random",
    seed=12345,
)
test_negative_samples = test_negative_sampler.get_negative_samples()


valset = SequenceDataset(
    mode="eval",
    mask_token=recsys_data.num_items + 1,
    u2seq=recsys_data.train_seqs,
    u2answer=recsys_data.val_seqs,
    max_len=max_len,
    negative_samples=test_negative_samples,
)

testset = SequenceDataset(
    mode="eval",
    mask_token=recsys_data.num_items + 1,
    u2seq=recsys_data.train_seqs,
    u2answer=recsys_data.test_seqs,
    max_len=max_len,
    negative_samples=test_negative_samples,
)

mymodel = BERTModel(
    hidden_size=256,
    num_items=recsys_data.num_items,  # item 的數量
    n_layers=2,
    dropout=0,
    heads=8,
    max_len=max_len,
)

# %%
train_loader = DataLoader(trainset, batch_size=12, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=12, shuffle=False, pin_memory=True)
test_loader = DataLoader(testset, batch_size=12, shuffle=False, pin_memory=True)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=LOG_PATH)

trainer = pl.Trainer(
    limit_train_batches=100, max_epochs=20, accelerator="gpu", logger=tb_logger
)
trainer.fit(mymodel, train_loader, val_loader)
trainer.test(mymodel, test_loader)
