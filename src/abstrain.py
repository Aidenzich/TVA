from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from src.config import DATA_PATH, LOG_PATH
from typing import Dict
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
import torch


class ABSTrain(metaclass=ABCMeta):
    @classmethod
    def fit(self, model, trainset, valset, config, testset=None):
        train_loader = DataLoader(
            trainset,
            batch_size=config["batch_size"],
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            valset,
            batch_size=config["batch_size"],
            shuffle=False,
            pin_memory=True,
        )

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=LOG_PATH)

        trainer = pl.Trainer(
            limit_train_batches=10,  # FIXME
            max_epochs=config["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=tb_logger,
        )
        trainer.fit(model, train_loader, val_loader)

        if testset != None:
            test_loader = DataLoader(
                testset,
                batch_size=config["batch_size"],
                shuffle=False,
                pin_memory=True,
            )
            trainer.test(model, test_loader)

    @abstractmethod
    def __init__(self):
        return NotImplemented
