# %%
import sys

sys.path.append("../")
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.model.BERT4Rec.model import BERTModel
from src.config import DATA_PATH, LOG_PATH
from src.train import train
import pickle


# 需要有一個地方來存放處理好的資料
with open(DATA_PATH / "data_cls.pkl", "rb") as f:
    recsys_data = pickle.load(f)


config = {
    "mask_prob": 0.15,
    "sample_size": 5000,
    "hidden_size": 256,
    "n_layers": 2,
    "dropout": 0,
    "head": 8,
    "max_len": 512,
    "batch_size": 12,
}


test_negative_sampler = NegativeSampler(
    train=recsys_data.train_seqs,
    val=recsys_data.val_seqs,
    test=recsys_data.test_seqs,
    user_count=recsys_data.num_users,
    item_count=recsys_data.num_items,
    sample_size=config["sample_size"],
    method="random",
    seed=12345,
)
test_negative_samples = test_negative_sampler.get_negative_samples()

trainset = SequenceDataset(
    mode="train",
    max_len=config["max_len"],
    mask_prob=config["mask_prob"],
    num_items=recsys_data.num_items,
    mask_token=recsys_data.num_items + 1,
    u2seq=recsys_data.train_seqs,
    seed=12345,
)

valset = SequenceDataset(
    mode="eval",
    max_len=config["max_len"],
    mask_token=recsys_data.num_items + 1,
    u2seq=recsys_data.train_seqs,
    u2answer=recsys_data.val_seqs,
    negative_samples=test_negative_samples,
)

testset = SequenceDataset(
    mode="eval",
    max_len=config["max_len"],
    mask_token=recsys_data.num_items + 1,
    u2seq=recsys_data.train_seqs,
    u2answer=recsys_data.test_seqs,
    negative_samples=test_negative_samples,
)

test_loader = DataLoader(
    testset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
)

mymodel = BERTModel(
    hidden_size=256,
    num_items=recsys_data.num_items,  # item 的數量
    n_layers=2,
    dropout=0,
    heads=8,
    max_len=config["max_len"],
)

train(
    model=mymodel,
    trainset=trainset,
    valset=valset,
    config=config,
    testset=testset,
    max_epochs=1,
    limit_batches=100,
)
