from src.datasets.matrix_dset import MatrixDataset
from src.models.VAECF.model import VAECFModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
from src.datasets.common import RecsysData
import numpy as np
import random


def _split_matrix_by_item(recdata: RecsysData):
    """
    Splitting matrix by random shuffle user for train, testing and validation
    """

    print("Splitting matrix by random shuffle user for train, testing and validation")

    # split to train val and test
    users = list(recdata.i2cat.values())

    random.shuffle(users)
    train_num = int(len(users) * 0.98)
    test_num = int(len(users) * 0.01)
    # val_num = len(users) - train_num - test_num
    # print(len(users[:train_num]))

    train_users = users[:train_num]
    test_users = users[-test_num:]
    val_users = users[train_num:-test_num]

    matrix = recdata.matrix.transpose()

    train_matrix = matrix[train_users, :]
    test_matrix = matrix[test_users, :]
    val_matrix = matrix[val_users, :]

    return train_matrix, test_matrix, val_matrix


def train(
    model_params: dict,
    trainer_config: dict,
    recdata: RecsysData,
    callbacks: list = [],
) -> None:

    train_matrix, test_matrix, val_matrix = _split_matrix_by_item(recdata)

    trainset = MatrixDataset(train_matrix)
    testset = MatrixDataset(test_matrix)
    valset = MatrixDataset(val_matrix)
    model = VAECFModel(
        hidden_dim=model_params["hidden_dim"],
        item_dim=recdata.num_users,
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
    matrix = recdata.matrix.transpose()

    inferset = MatrixDataset(matrix)
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

    with open(latent_factor_path / "decode_result.npy", "wb+") as f:
        np.save(f, all_y)

    with open(latent_factor_path / "encode_result.npy", "wb+") as f:
        np.save(f, all_z)

    return predict_result
