import pickle

from torch.utils.data import DataLoader

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from ray import tune, air
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from src.datasets.seq_dset import SequenceDataset
from src.models.BERT4Rec.model import BERTModel
from src.datasets.negative_sampler import NegativeSampler
from src.configs import DATA_PATH, LOG_PATH


def train_tune2(config):
    with open(DATA_PATH / "data_cls.pkl", "rb") as f:
        recsys_data = pickle.load(f)
    trainset = SequenceDataset(
        mode="train",
        max_len=512,
        mask_prob=0.15,
        num_items=recsys_data.num_items,
        mask_token=recsys_data.num_items + 1,
        u2seq=recsys_data.train_seqs,
        seed=12345,
    )

    valset = SequenceDataset(
        mode="eval",
        max_len=512,
        mask_token=recsys_data.num_items + 1,
        u2seq=recsys_data.train_seqs,
        u2answer=recsys_data.val_seqs,
        negative_samples=test_negative_samples,
        seed=12345,
    )
    epochs = 5

    early_stop_callback = EarlyStopping(
        monitor="recall@1", patience=5, verbose=False, mode="max"
    )

    tune_callback = TuneReportCallback(
        {"loss": "train_loss", "recall": "recall@1"}, on="validation_end"
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=tune.get_trial_dir())
    accelerator = "gpu"
    trainer = Trainer(
        callbacks=[early_stop_callback, tune_callback],
        logger=tb_logger,
        max_epochs=epochs,
        accelerator=accelerator,
        limit_predict_batches=0.01,
    )

    model = BERTModel(
        num_items=recsys_data.num_items,
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        heads=config["head"],
        max_len=config["max_len"],
    )

    train_loader = DataLoader(trainset, batch_size=6, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=6, shuffle=False, pin_memory=True)
    trainer.fit(model, train_loader, val_loader)


def tune_bert4rec(config):

    resources_per_trial = {"cpu": 12, "gpu": 1}
    # accelerator = "cuda" if resources_per_trial["gpu"] > 0 else "cpu"
    analysis = tune.Tuner(
        tune.with_resources(train_tune2, resources_per_trial),
        tune_config=tune.TuneConfig(metric="recall", mode="max", num_samples=1),
        param_space=config,
        run_config=air.RunConfig(name="tune_bert4rec", local_dir=LOG_PATH),
    ).fit()

    print("Best hyperparameters found were: ", analysis.get_best_result().config)


if __name__ == "__main__":
    config = {
        "hidden_size": tune.choice([64, 128, 256]),
        "n_layers": tune.choice([2, 3]),
        "dropout": tune.choice([0, 0.1, 0.2]),
        # "max_len": tune.choice([64, 128, 256, 512]),
        "max_len": 512,
        "head": tune.choice([2, 4, 8, 16]),
    }

    with open(DATA_PATH / "data_cls.pkl", "rb") as f:
        recsys_data = pickle.load(f)

    test_negative_sampler = NegativeSampler(
        train=recsys_data.train_seqs,
        val=recsys_data.val_seqs,
        test=recsys_data.test_seqs,
        user_count=recsys_data.num_users,
        item_count=recsys_data.num_items,
        sample_size=128,
        method="popular",
        seed=12345,
    )
    test_negative_samples = test_negative_sampler.get_negative_samples()

    tune_bert4rec(config)
