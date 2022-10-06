from torch.utils.data import DataLoader

from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.model.BERT4Rec.model import BERTModel
from src.config import DATA_PATH
from src.train import train
import pickle
import json


def config_adapter(params_config):
    print("\033[93m" + json.dumps(params_config, sort_keys=True, indent=4) + "\033[0m")
    with open(DATA_PATH / "data_cls.pkl", "rb") as f:
        recsys_data = pickle.load(f)

    test_negative_sampler = NegativeSampler(
        train=recsys_data.train_seqs,
        val=recsys_data.val_seqs,
        test=recsys_data.test_seqs,
        user_count=recsys_data.num_users,
        item_count=recsys_data.num_items,
        sample_size=params_config["sample_size"],
        method="random",
        seed=12345,
    )
    test_negative_samples = test_negative_sampler.get_negative_samples()

    trainset = SequenceDataset(
        mode="train",
        max_len=params_config["max_len"],
        mask_prob=params_config["mask_prob"],
        num_items=recsys_data.num_items,
        mask_token=recsys_data.num_items + 1,
        u2seq=recsys_data.train_seqs,
        seed=12345,
    )

    valset = SequenceDataset(
        mode="eval",
        max_len=params_config["max_len"],
        mask_token=recsys_data.num_items + 1,
        u2seq=recsys_data.train_seqs,
        u2answer=recsys_data.val_seqs,
        negative_samples=test_negative_samples,
    )

    testset = SequenceDataset(
        mode="eval",
        max_len=params_config["max_len"],
        mask_token=recsys_data.num_items + 1,
        u2seq=recsys_data.train_seqs,
        u2answer=recsys_data.test_seqs,
        negative_samples=test_negative_samples,
    )

    test_loader = DataLoader(
        testset,
        batch_size=params_config["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    model = BERTModel(
        hidden_size=256,
        num_items=recsys_data.num_items,  # item 的數量
        n_layers=2,
        dropout=0,
        heads=8,
        max_len=params_config["max_len"],
    )

    train(
        model=model,
        trainset=trainset,
        valset=valset,
        config=params_config,
        testset=testset,
        max_epochs=1,
        limit_batches=100,
    )
