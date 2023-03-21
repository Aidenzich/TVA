from src.datasets.matrix_dset import MatrixDataset
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
    trainset = MatrixDataset(recdata.train_matrix)
    testset = MatrixDataset(recdata.test_matrix)
    valset = MatrixDataset(recdata.val_matrix)
    model = VAECFModel(
        hidden_dim=model_params["hidden_dim"],
        item_dim=recdata.num_items,
        act_fn=model_params["act_fn"],
        autoencoder_structure=model_params["autoencoder_structure"],
        likelihood=model_params["likelihood"],
        beta=model_params["beta"],
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
    inferset = MatrixDataset(recdata.matrix)
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

    with open(latent_factor_path / (recdata.filename + "_variance.npy"), "wb+") as f:
        np.save(f, all_y)

    with open(
        latent_factor_path / (recdata.filename + "_latent_factor.npy"), "wb+"
    ) as f:
        np.save(f, all_z)

    return predict_result
