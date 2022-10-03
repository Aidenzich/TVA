# %%
import random
from ray import tune
from pytorch_lightning import loggers as pl_loggers

from ray.tune.integration.pytorch_lightning import TuneReportCallback
import pickle

from torch.utils.data import DataLoader

from src.dataset.seq_dset import SequenceDataset
from src.model.BERT4Rec.model import BERTModel
from src.config import DATA_PATH
from src.model.BERT4Rec.negative_sampler import NegativeSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from pytorch_lightning import Trainer
from ray.tune import CLIReporter

#%%


def train_tune(config, epochs=5, accelerator="cpu"):

    early_stop_callback = EarlyStopping(
        monitor="Recall@1", patience=5, verbose=False, mode="max"
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=tune.get_trial_dir())
    tune_callback = TuneReportCallback(
        {"loss": "train_loss", "recall": "Recall@1"}, on="validation_end"
    )

    trainer = Trainer(
        callbacks=[early_stop_callback, tune_callback],
        logger=tb_logger,
        max_epochs=epochs,
        accelerator=accelerator,
    )

    with open(DATA_PATH / "data_cls.pkl", "rb") as f:
        recsys_data = pickle.load(f)

    model = BERTModel(
        num_items=recsys_data.num_items,
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        heads=8,
        max_len=config["max_len"],
    )

    trainset = SequenceDataset(
        mode="train",
        max_len=config["max_len"],
        mask_prob=0.15,
        num_items=recsys_data.num_items,
        mask_token=recsys_data.num_items + 1,
        u2seq=recsys_data.train_seqs,
        rng=random.Random(12345),
    )

    test_negative_sampler = NegativeSampler(
        train=recsys_data.train_seqs,
        val=recsys_data.val_seqs,
        test=recsys_data.test_seqs,
        user_count=recsys_data.num_users,
        item_count=recsys_data.num_items,
        sample_size=100000,
        method="popular",
        seed=12345,
    )
    test_negative_samples = test_negative_sampler.get_negative_samples()

    valset = SequenceDataset(
        mode="eval",
        mask_token=recsys_data.num_items + 1,
        u2seq=recsys_data.train_seqs,
        u2answer=recsys_data.val_seqs,
        max_len=config["max_len"],
        negative_samples=test_negative_samples,
    )

    train_loader = DataLoader(trainset, batch_size=6, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=6, shuffle=False, pin_memory=True)
    trainer.fit(model, train_loader, val_loader)


def tune_bert4rec():
    config = {
        "hidden_size": tune.choice([64, 128, 256]),
        "n_layers": tune.choice([2, 3]),
        "dropout": tune.choice([0, 0.1, 0.2]),
        "max_len": tune.choice([64, 128, 256, 512]),
    }
    # config = {
    #     "hidden_size": 128,
    #     "n_layers": 2,
    #     "dropout": 0.1,
    #     "max_len": 64,
    # }

    reporter = CLIReporter(
        parameter_columns=["hidden_size", "n_layers", "dropout", "max_len"],
        metric_columns=["recall"],
    )
    resources_per_trial = {"cpu": 4, "gpu": 1}
    accelerator = "cuda" if resources_per_trial["gpu"] > 0 else "cpu"
    # scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)
    train_fn_with_parameters = tune.with_parameters(
        train_tune, epochs=10, accelerator=accelerator
    )

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        progress_reporter=reporter,
        # scheduler=scheduler,
        metric="recall",
        num_samples=1,  # trials number
        mode="max",
        config=config,
    )

    print("Best hyperparameters found were: ", analysis.best_config)


# %%
tune_bert4rec()

# %%
# with open(DATA_PATH / "data_cls.pkl", "rb") as f:
#     recsys_data = pickle.load(f)


# recsys_data
# # %%
# recsys_data.num_items

# #%%
# recsys_data.num_users