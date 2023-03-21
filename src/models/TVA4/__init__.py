from torch import ne
from src.datasets.tvaseq_dset import TVASequenceDataset

# from src.datasets.negative_sampler import NegativeSampler
from .model import TVAModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np


def train(model_params, trainer_config, recdata, callbacks=[]):
    user_latent_factor = np.load(model_params["user_latent_factor"]["path"])

    item_latent_factor = np.load(model_params["user_latent_factor"]["path"])

    trainset = TVASequenceDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        seed=trainer_config["seed"],
        u2timeseq=recdata.train_timeseqs,
        latent_factor=user_latent_factor,
        item_latent_factor=item_latent_factor,
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
        latent_factor=user_latent_factor,
        seed=trainer_config["seed"],
        item_latent_factor=item_latent_factor,
    )

    u2seqs_for_test = {}
    u2timeseqs_for_test = {}
    for u in recdata.users_seqs:
        # Remove the last item of the fully user's sequence
        u2seqs_for_test[u] = recdata.users_seqs[u][:-1]
        # Remove the last and first item of the fully user's time sequence
        u2timeseqs_for_test[u] = recdata.users_timeseqs[u][:-1]

    testset = TVASequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=u2seqs_for_test,
        u2answer=recdata.test_seqs,
        u2timeseq=u2timeseqs_for_test,
        num_items=recdata.num_items,
        u2eval_time=recdata.test_timeseqs,
        latent_factor=user_latent_factor,
        seed=trainer_config["seed"],
        item_latent_factor=item_latent_factor,
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


def infer(ckpt_path, recdata, rec_ks=10):
    """rec k is the number of items to recommend"""
    ##### INFER ######
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

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

    variance = np.load(CACHE_PATH / "variance.npy")

    inferset = TVASequenceDataset(
        mode="inference",
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        max_len=model.max_len,
        vae_matrix=variance,
    )

    infer_loader = DataLoader(inferset, batch_size=4, shuffle=False, pin_memory=True)

    model.to(device)
    predict_result: dict = {}
    with torch.no_grad():
        for batch in tqdm(infer_loader):
            seqs, vae_squence, candidates, users = batch
            seqs, vae_squence, candidates, users = (
                seqs.to(device),
                vae_squence.to(device),
                candidates.to(device),
                users.to(device),
            )

            scores = model(seqs, vae_squence)
            scores = scores[:, -1, :]  # B x V
            scores = scores.gather(1, candidates)  # B x C
            rank = (-scores).argsort(dim=1)
            predict = candidates.gather(1, rank)

            predict_dict = {
                u: predict[idx].tolist()[:rec_ks]
                for idx, u in enumerate(users.cpu().numpy().flatten())
            }
            predict_result.update(predict_dict)

    return predict_result
