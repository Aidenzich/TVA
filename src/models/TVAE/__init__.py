from torch import ne
from src.datasets.tvae_dset import TVASequenceDataset

# from src.datasets.negative_sampler import NegativeSampler
from .model import TVAModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np
from tqdm import tqdm
from ..BERT4RecS import get_slidewindow


def train(model_params, trainer_config, recdata, callbacks=[]):
    slided_u2train_seqs = {}

    for u in tqdm(recdata.train_seqs):
        slided_user_seqs = get_slidewindow(
            recdata.train_seqs[u], model_params["max_len"], step=1
        )

        for idx, seqs in enumerate(slided_user_seqs):
            slided_u2train_seqs[str(u) + "." + str(idx)] = seqs

    print(f"Before sliding window data num: {len(recdata.train_seqs)}")
    print(f"After sliding window data num: {len(slided_u2train_seqs)}")

    trainset = TVASequenceDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=slided_u2train_seqs,
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

    testset = TVASequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        u2val=recdata.val_seqs,
        u2timeseq=recdata.train_timeseqs,
        num_items=recdata.num_items,
        u2eval_time=recdata.test_timeseqs,
        seed=trainer_config["seed"],
        user_matrix=recdata.matrix,
    )

    model = TVAModel(
        num_items=recdata.num_items,
        model_params=model_params,
        trainer_config=trainer_config,
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
