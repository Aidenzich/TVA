from src.datasets.matrix_dset import MatrixDataset
from src.models.VAECF.model import VAECF
from src.adapters.lightning_adapter import fit


def vaecf_train(
    model_params: dict,
    trainer_config: dict,
    recdata,
    callbacks: list = [],
):
    trainset = MatrixDataset(recdata.matrix)
    model = VAECF(
        k=model_params["k"],
        item_dim=recdata.num_items,
        act_fn=model_params["act_fn"],
        autoencoder_structure=model_params["autoencoder_structure"],
        likelihood=model_params["likelihood"],
        beta=model_params["beta"],
    )

    fit(
        model=model,
        trainset=trainset,
        valset=None,
        trainer_config=trainer_config,
        model_params=model_params,
        testset=None,
        callbacks=callbacks,
    )
