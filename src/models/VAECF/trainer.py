from src.datasets.matrix_dset import MatrixDataset
from src.models.VAECF.model import VAECF
from src.adapters.lightning_adapter import fit


def vaecf_train(
    model_params: dict,
    trainer_config: dict,
    recdata,
    callbacks: list = [],
):
    trainset = MatrixDataset(recdata.train_matrix)
    testset = MatrixDataset(recdata.test_matrix)
    valset = MatrixDataset(recdata.val_matrix)
    model = VAECF(
        hidden_dim=model_params["hidden_dim"],
        item_dim=recdata.num_items,
        act_fn=model_params["act_fn"],
        autoencoder_structure=model_params["autoencoder_structure"],
        likelihood=model_params["likelihood"],
        beta=model_params["beta"],
    )

    print(model_params)

    fit(
        model=model,
        trainset=trainset,
        valset=valset,
        trainer_config=trainer_config,
        model_params=model_params,
        testset=None,
        callbacks=callbacks,
    )
