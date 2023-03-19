from torch import ne
from src.datasets.sasrec_dset import SASRecDataset
from src.datasets.negative_sampler import NegativeSampler
from .model import SASRecModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np


def train(model_params, trainer_config, recdata, callbacks=[]):
    test_negative_sampler = NegativeSampler(
        train=recdata.train_seqs,
        val=recdata.val_seqs,
        test=recdata.test_seqs,
        item_count=recdata.num_items,
        sample_size=trainer_config["sample_size"],
        method=trainer_config["sample_method"],
        seed=trainer_config["seed"],
        dataclass_name=recdata.filename,
    )

    test_negative_samples = test_negative_sampler.get_negative_samples()

    trainset = SASRecDataset(
        mode="train",
        max_len=model_params["max_len"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        negative_samples=test_negative_samples,
        seed=trainer_config["seed"],
    )

    valset = SASRecDataset(
        mode="eval",
        max_len=model_params["max_len"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        negative_samples=test_negative_samples,
        seed=trainer_config["seed"],
    )

    testset = SASRecDataset(
        mode="eval",
        max_len=model_params["max_len"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        negative_samples=test_negative_samples,
        seed=trainer_config["seed"],
    )

    model = SASRecModel(
        num_items=recdata.num_items,
        num_users=recdata.num_users,
        model_params=model_params,
    )

    fit(
        model=model,
        trainset=trainset,
        valset=valset,
        trainer_config=trainer_config,
        model_params=model_params,
        callbacks=callbacks,
    )

    # NOTE: RuntimeError: CUDA error: device-side assert triggered
    # might be caused by the embedding layer
