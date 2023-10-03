from src.datasets.base import RecsysData
from src.datasets.tva_dset import TVASequenceDataset
from .model import TVAModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np
from ..BERT4RecS import get_slidewindow
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model_params: Dict, trainer_config: Dict, recdata: RecsysData, callbacks: list = []
) -> None:
    user_latent_factor = None
    item_latent_factor = None
    if model_params.get("user_latent_factor", None) is not None:
        user_latent_factor = np.load(model_params["user_latent_factor"]["path"])

    if model_params.get("item_latent_factor", None) is not None:
        item_latent_factor = np.load(model_params["item_latent_factor"]["path"])

    # Sliding window
    use_sliding_window = trainer_config.get("sliding_window", False)
    sliding_step = trainer_config.get("sliding_step", 1)

    train_seqs = recdata.train_seqs
    train_timeseqs = recdata.train_timeseqs

    if use_sliding_window:
        print("Sliding window is enabled. Handling data...")
        slided_u2train_seqs = {}
        slided_u2time_seqs = {}

        for u in tqdm(recdata.train_seqs):
            slided_user_seqs = get_slidewindow(
                recdata.train_seqs[u], model_params["max_len"], step=sliding_step
            )
            slided_time_seqs = get_slidewindow(
                recdata.train_timeseqs[u], model_params["max_len"], step=sliding_step
            )

            for idx, seqs in enumerate(slided_user_seqs):
                save_key = str(u) + "." + str(idx)
                slided_u2train_seqs[save_key] = seqs
                slided_u2time_seqs[save_key] = slided_time_seqs[idx]

        train_seqs = slided_u2train_seqs
        train_timeseqs = slided_u2time_seqs

        print(f"Before sliding window data num: {len(recdata.train_seqs)}")
        print(f"After sliding window data num: {len(slided_u2train_seqs)}")

    trainset = TVASequenceDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=train_seqs,
        u2timeseq=train_timeseqs,
        user_latent_factor=user_latent_factor,
        item_latent_factor=item_latent_factor,
        seed=trainer_config["seed"],
        num_mask=model_params.get("num_mask", 1),
    )

    valset = TVASequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        u2timeseq=recdata.train_timeseqs,
        num_items=recdata.num_items,
        u2answer_time=recdata.val_timeseqs,
        user_latent_factor=user_latent_factor,
        item_latent_factor=item_latent_factor,
        seed=trainer_config["seed"],
    )

    testset = TVASequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        u2answer_time=recdata.test_timeseqs,
        u2timeseq=recdata.train_timeseqs,
        num_items=recdata.num_items,
        # Add val item into item seqence
        u2val=recdata.val_seqs,
        u2val_time=recdata.val_timeseqs,
        user_latent_factor=user_latent_factor,
        seed=trainer_config["seed"],
        item_latent_factor=item_latent_factor,
    )

    model = TVAModel(
        num_items=recdata.num_items,
        trainer_config=trainer_config,
        model_params=model_params,
        data_class=recdata.filename,
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


def infer(ckpt_path, recdata: RecsysData, rec_ks: int = 10) -> Dict:
    """rec k is the number of items to recommend"""
    ##### INFER ######
    import torch

    device = torch.device("cuda:0")

    torch.cuda.empty_cache()

    model = TVAModel.load_from_checkpoint(ckpt_path)

    sample_num = int(recdata.num_items * 0.2)

    if sample_num > 10000:
        print(
            "Sample num is too large, set to 10000. (Due to 2070's memory limitation)"
        )
        sample_num = 10000

    samples = {}

    sample_items = (
        recdata.dataframe["item_id"].value_counts().index.tolist()[:sample_num]
    )
    for u in range(recdata.num_users):
        samples[u] = sample_items

    # TODO OUT OF DATE
    # variance = np.load(CACHE_PATH / "variance.npy")

    # inferset = TVASequenceDataset(
    #     mode="inference",
    #     mask_token=recdata.num_items + 1,
    #     u2seq=recdata.train_seqs,
    #     u2answer=recdata.val_seqs,
    #     max_len=model.max_len,
    #     vae_matrix=variance,
    # )

    # infer_loader = DataLoader(inferset, batch_size=4, shuffle=False, pin_memory=True)

    # model.to(device)
    # predict_result: dict = {}
    # with torch.no_grad():
    #     for batch in tqdm(infer_loader):
    #         seqs, vae_squence, candidates, users = batch
    #         seqs, vae_squence, candidates, users = (
    #             seqs.to(device),
    #             vae_squence.to(device),
    #             candidates.to(device),
    #             users.to(device),
    #         )

    #         scores = model(seqs, vae_squence)
    #         scores = scores[:, -1, :]  # B x V
    #         scores = scores.gather(1, candidates)  # B x C
    #         rank = (-scores).argsort(dim=1)
    #         predict = candidates.gather(1, rank)

    #         predict_dict = {
    #             u: predict[idx].tolist()[:rec_ks]
    #             for idx, u in enumerate(users.cpu().numpy().flatten())
    #         }
    #         predict_result.update(predict_dict)

    return {}
