from src.adapters.lightning_adapter import fit
from src.datasets.cvae_dset import CVAEDataset
from .model import SASRecModel


def train(model_params, trainer_config, recdata, callbacks=[]) -> None:
    trainset = CVAEDataset(
        mode="train",
        u2seq=recdata.users_seqs,
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        max_len=model_params["max_len"],
    )

    valset = CVAEDataset(
        mode="valid",
        u2seq=recdata.users_seqs,
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        max_len=model_params["max_len"],
    )

    testset = CVAEDataset(
        mode="test",
        u2seq=recdata.users_seqs,
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        max_len=model_params["max_len"],
    )

    model = SASRecModel(
        num_items=recdata.num_items,
        model_params=model_params,
        trainer_config=trainer_config,
    )

    fit(
        model=model,
        trainset=trainset,
        valset=valset,
        testset=testset,
        trainer_config=trainer_config,
        model_params=model_params,
        callbacks=callbacks,
    )
