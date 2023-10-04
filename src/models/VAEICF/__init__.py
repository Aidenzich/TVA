from src.datasets.vaecf_dset import VAECFDataset, split_random_matrix_by_item
from src.models.VAECF.model import VAECFModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
from src.datasets.base import RecsysData
import numpy as np


def train(
    model_params: dict,
    trainer_config: dict,
    recdata: RecsysData,
    callbacks: list = [],
) -> None:
    vae_split_type = model_params["split_type"]

    if vae_split_type == "loo":
        transaction = recdata.dataframe
        # Group by item and get the last two user buy the item
        transaction.sort_values(by=["item_id", "timestamp"], inplace=True)
        test_transaction = transaction.groupby("item_id").tail(1).sort_values("item_id")

        # Convert to dictionary like itemid: [userid]
        test_user_seqs = (
            test_transaction.groupby("item_id")["user_id"].apply(list).to_dict()
        )

        val_transaction = transaction.groupby("item_id").tail(2).sort_values("item_id")

        # Remove test_transaction row from val_transaction
        val_transaction = val_transaction.drop(test_transaction.index)

        val_user_seqs = (
            val_transaction.groupby("item_id")["user_id"].apply(list).to_dict()
        )
        trainset = VAECFDataset(
            recdata.matrix.transpose(),
            mode="train",
            split_type="loo",
        )
        valset = VAECFDataset(
            recdata.matrix.transpose(),
            u2val=val_user_seqs,
            mode="eval",
            split_type="loo",
        )
        testset = VAECFDataset(
            recdata.test_matrix.transpose(),
            u2val=test_user_seqs,
            mode="eval",
            split_type="loo",
        )
    elif vae_split_type == "random":
        train_matrix, test_matrix, val_matrix = split_random_matrix_by_item(recdata)
        trainset = VAECFDataset(
            train_matrix,
            split_type="random",
        )
        valset = VAECFDataset(
            test_matrix,
            split_type="random",
        )
        testset = VAECFDataset(
            val_matrix,
            split_type="random",
        )

    model = VAECFModel(
        num_items=recdata.num_users,
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
    latent_factor_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    matrix = recdata.matrix.transpose()
    # matrix = recdata.test_matrix.transpose()

    inferset = VAECFDataset(matrix, split_type="")
    model = VAECFModel.load_from_checkpoint(ckpt_path)
    model.to(device)

    infer_loader = DataLoader(
        inferset,
        batch_size=3096,
        shuffle=False,
        pin_memory=True,
    )

    predict_result: dict = {}
    user_count = 0

    all_y = None
    all_z_u = None

    with torch.no_grad():
        for batch in tqdm(infer_loader):
            batch = batch["matrix"].to(device)

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

    return False  # Return False to prevent the reverse_id
