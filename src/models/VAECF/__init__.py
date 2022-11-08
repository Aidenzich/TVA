from src.datasets.matrix_dset import MatrixDataset
from src.models.VAECF.model import VAECFModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np


def train_vaecf(
    model_params: dict,
    trainer_config: dict,
    recdata,
    callbacks: list = [],
):
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

    # print(model_params)

    fit(
        model=model,
        trainset=trainset,
        valset=valset,
        trainer_config=trainer_config,
        model_params=model_params,
        testset=testset,
        callbacks=callbacks,
    )


def infer_vaecf(ckpt_path, recdata, rec_ks=100):
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm

    device = torch.device("cuda:0")
    inferset = MatrixDataset(recdata.matrix)
    model = VAECFModel.load_from_checkpoint(ckpt_path)
    infer_loader = DataLoader(inferset, batch_size=2048, shuffle=False, pin_memory=True)
    model.to(device)
    predict_result: dict = {}
    user_count = 0

    all_y = None
    all_z = None

    with torch.no_grad():
        for batch in tqdm(infer_loader):
            batch = batch.to(device)
            z_u, _ = model.vae.encode(batch)
            y = model.vae.decode(z_u)
            print(z_u.shape)
            # seen = batch != 0
            # y[seen] = 0

            top_k = y.topk(rec_ks, dim=1)[1]
            if all_y is None:
                all_y = y.cpu().numpy().astype(np.float16)
                all_z = z_u.cpu().numpy().astype(np.float16)
            else:
                all_y = np.concatenate([all_y, y.cpu().numpy().astype(np.float16)])
                all_z = np.concatenate([all_z, z_u.cpu().numpy().astype(np.float16)])
            for i in range(batch.shape[0]):
                predict_result[user_count] = top_k[i].tolist()
                user_count = user_count + 1

    with open(CACHE_PATH / "variance.npy", "wb") as f:
        np.save(f, all_y)

    with open(CACHE_PATH / "latent_factor.npy", "wb") as f:
        np.save(f, all_z)

    return predict_result
