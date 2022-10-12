from src.datasets.matrix_dset import MatrixDataset
from src.models.VAECF.model import VAECFModel
from src.adapters.lightning_adapter import fit


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
        testset=None,
        callbacks=callbacks,
    )


def infer_vaecf(model_path, recdata, rec_ks=10):
    from torch.utils.data import DataLoader
    import torch
    from tqdm import tqdm

    device = torch.device("cuda:0")
    inferset = MatrixDataset(recdata.matrix)
    model = VAECFModel.load_from_checkpoint(model_path)
    infer_loader = DataLoader(inferset, batch_size=12, shuffle=False, pin_memory=True)
    model.to(device)
    predict_result: dict = {}
    user_count = 0

    with torch.no_grad():
        for batch in tqdm(infer_loader):
            batch = batch.to(device)
            z_u, _ = model.vae.encode(batch)
            y = model.vae.decode(z_u)
            seen = batch != 0
            y[seen] = 0
            top_k = y.topk(rec_ks, dim=1)[1]

            for i in range(batch.shape[0]):
                predict_result[user_count] = top_k[i].tolist()
                user_count = user_count + 1

    return predict_result
