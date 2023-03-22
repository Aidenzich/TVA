from torch import ne
from src.datasets.tvae_dset import TVASequenceDataset

# from src.datasets.negative_sampler import NegativeSampler
from .model import TVAModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np


def train(model_params, trainer_config, recdata, callbacks=[]):

    trainset = TVASequenceDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        seed=trainer_config["seed"],
        u2timeseq=recdata.train_timeseqs,
        user_matrix=recdata.matrix,
    )

    valset = TVASequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        u2timeseq=recdata.train_timeseqs,
        num_items=recdata.num_items,
        u2eval_time=recdata.val_timeseqs,
        seed=trainer_config["seed"],
        user_matrix=recdata.matrix,
    )

    # u2seqs_for_test = {}
    # for u in recdata.users_seqs:
    #     # Remove the last and first item of the fully user's sequence
    #     u2seqs_for_test[u] = recdata.users_seqs[u][:-1]

    testset = TVASequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        u2timeseq=recdata.train_timeseqs,
        num_items=recdata.num_items,
        u2eval_time=recdata.test_timeseqs,
        seed=trainer_config["seed"],
        user_matrix=recdata.matrix,
    )

    model = TVAModel(
        num_items=recdata.num_items,
        model_params=model_params,
    )

    fit(
        model=model,
        trainset=trainset,
        valset=valset,
        trainer_config=trainer_config,
        model_params=model_params,
        testset=testset,
        callbacks=callbacks,
    )
