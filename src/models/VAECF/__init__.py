from src.datasets.vaecf_dset import VAECFDataset, split_random_matrix_by_user
from src.models.VAECF.model import VAECFModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np


def train(
    model_params: dict,
    trainer_config: dict,
    recdata,
    callbacks: list = [],
) -> None:
    vae_split_type = model_params["split_type"]
    if vae_split_type == "loo":
        trainset = VAECFDataset(recdata.matrix)

        valset = VAECFDataset(recdata.matrix, u2val=recdata.val_seqs, mode="eval")

        testset = VAECFDataset(
            recdata.test_matrix, u2val=recdata.test_seqs, mode="eval"
        )

    elif vae_split_type == "random":
        train_matrix, test_matrix, val_matrix = split_random_matrix_by_user(recdata)
        trainset = VAECFDataset(
            train_matrix,
            split_type="random",
        )
        testset = VAECFDataset(
            test_matrix,
            split_type="random",
        )
        valset = VAECFDataset(
            val_matrix,
            split_type="random",
        )

    model = VAECFModel(
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


def infer(ckpt_path, recdata, rec_ks=100):
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm

    latent_factor_path = ckpt_path.parent.parent / "latent_factor/"
    latent_factor_path.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda:0")
    inferset = VAECFDataset(recdata.test_matrix)
    model = VAECFModel.load_from_checkpoint(ckpt_path)
    infer_loader = DataLoader(inferset, batch_size=3096, shuffle=False, pin_memory=True)
    model.to(device)
    predict_result: dict = {}
    user_count = 0

    all_y = None
    all_z_u = None

    with torch.no_grad():
        for batch in tqdm(infer_loader):
            batch = batch.to(device)
            z_u, z_sigma = model.vae.encode(batch)
            y = model.vae.decode(z_u)

            z_u, z_sigma = z_u.cpu().numpy().astype(
                np.float16
            ), z_sigma.cpu().numpy().astype(np.float16)

            top_k = y.topk(rec_ks, dim=1)[1]
            # seen = batch != 0
            # y[seen] = 0

            y = y.cpu().numpy().astype(np.float16)

            # Saving latent factor
            if all_y is None:
                all_y, all_z_u, all_z_sigma = y, z_u, z_sigma
            else:
                all_y = np.concatenate([all_y, y])
                all_z_u = np.concatenate([all_z_u, z_u])
                all_z_sigma = np.concatenate([all_z_sigma, z_sigma])

            for i in range(batch.shape[0]):
                predict_result[user_count] = top_k[i].tolist()
                user_count = user_count + 1

    all_z = np.concatenate([all_z_u, all_z_sigma], axis=1)

    with open(latent_factor_path / "decode_result.npy", "wb+") as f:
        np.save(f, all_y)

    with open(latent_factor_path / "encode_result.npy", "wb+") as f:
        np.save(f, all_z)

    return predict_result
