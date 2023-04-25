from src.datasets.cbit_dset import CBiTDataset
from .model import CBiTModel
from ..BERT4RecS import get_slidewindow
from src.adapters.lightning_adapter import fit
from tqdm import tqdm


def train(model_params, trainer_config, recdata, callbacks=[]) -> None:
    slided_u2train_seqs = {}

    for u in tqdm(recdata.train_seqs):
        slided_user_seqs = get_slidewindow(
            recdata.train_seqs[u], model_params["max_len"], step=1
        )

        for idx, seqs in enumerate(slided_user_seqs):
            slided_u2train_seqs[str(u) + "." + str(idx)] = seqs

    print(f"Before sliding window data num: {len(recdata.train_seqs)}")
    print(f"After sliding window data num: {len(slided_u2train_seqs)}")

    trainset = CBiTDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=slided_u2train_seqs,
        seed=trainer_config["seed"],
        num_positive=model_params["num_positive"],
    )

    valset = CBiTDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        num_positive=model_params["num_positive"],
    )

    testset = CBiTDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        u2seq=recdata.train_seqs,
        u2val=recdata.val_seqs,
        u2answer=recdata.test_seqs,
        num_positive=model_params["num_positive"],
    )

    model = CBiTModel(
        num_items=recdata.num_items,
        model_params=model_params,
        trainer_config=trainer_config,
        data_class=recdata.filename
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
