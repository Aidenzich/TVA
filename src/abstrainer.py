from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from src.config import DATA_PATH, LOG_PATH
from typing import Dict
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch


class ABSTrainer(metaclass=ABCMeta):
    @staticmethod
    def fit(model, trainset, valset, trainer_config, model_params, testset=None, callbacks=[]):
        tune_config = trainer_config.get("tune")
        if tune_config:
            tune_callback = TuneReportCallback(
                tune_config["report"] , on="validation_end"
            )
            callbacks.append(tune_callback)
        
        early_stop_config = trainer_config.get("early_stopping")
        if trainer_config.get("early_stopping"):
            early_stop_callback = EarlyStopping(
                monitor=early_stop_config.get("monitor"), 
                patience=early_stop_config.get("patience"), 
                verbose=early_stop_config.get("verbose"), 
                mode=early_stop_config.get("mode")
            )
            callbacks.append(early_stop_callback)
                    
        train_loader = DataLoader(
            trainset,
            batch_size=model_params["batch_size"],
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            valset,
            batch_size=model_params["batch_size"],
            shuffle=False,
            pin_memory=True,
        )

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=LOG_PATH)

        trainer = pl.Trainer(
            callbacks=callbacks,
            limit_train_batches=10,  # FIXME
            max_epochs=trainer_config["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=tb_logger,
        )
        trainer.fit(model, train_loader, val_loader)

        if testset != None:
            test_loader = DataLoader(
                testset,
                batch_size=model_params["batch_size"],
                shuffle=False,
                pin_memory=True,
            )
            trainer.test(model, test_loader)

    @abstractmethod
    def __init__(self):
        return NotImplemented

    @abstractmethod
    def train():
        return NotImplemented
    
    @abstractmethod
    def tuner():
        return NotImplemented
    