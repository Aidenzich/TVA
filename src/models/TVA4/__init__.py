from torch import ne
from src.datasets.vaeseq_dset import VAESequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from .model import TVAModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np


def train_tva2(model_params, trainer_config, recdata, callbacks=[]):
    latent_factor = np.load(CACHE_PATH / (recdata.filename + "_latent_factor.npy"))

    # FIXME This can be store in the RecData class
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

    trainset = VAESequenceDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        seed=trainer_config["seed"],
        u2timeseq=recdata.train_timeseqs,
        latent_factor=latent_factor,
    )

    valset = VAESequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        negative_samples=test_negative_samples,
        u2timeseq=recdata.train_timeseqs,
        latent_factor=latent_factor,
        seed=trainer_config["seed"],
    )

    testset = VAESequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        negative_samples=test_negative_samples,
        u2timeseq=recdata.train_timeseqs,
        latent_factor=latent_factor,
        seed=trainer_config["seed"],
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


def infer_tva2(ckpt_path, recdata, rec_ks=10, negative_samples=None):
    """rec k is the number of items to recommend"""
    ##### INFER ######
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = torch.device("cuda:0")

    torch.cuda.empty_cache()

    model = TVAModel.load_from_checkpoint(ckpt_path)

    if negative_samples == None:
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

    inferset = VAESequenceDataset(
        mode="inference",
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        max_len=model.max_len,
        negative_samples=samples,
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
