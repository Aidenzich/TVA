from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from src.config import DATA_PATH, LOG_PATH
from typing import Dict
from torch.utils.data import Dataset


def train(
    model: pl.LightningModule,
    trainset: Dataset,
    valset: Dataset,
    config: Dict,
    max_epochs: int,
    limit_batches: int = 100,
    testset: int = None,
):
    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        valset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=LOG_PATH)

    trainer = pl.Trainer(
        limit_train_batches=limit_batches,
        max_epochs=max_epochs,
        accelerator="gpu",
        logger=tb_logger,
    )
    trainer.fit(model, train_loader, val_loader)

    if testset != None:
        test_loader = DataLoader(
            testset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
        )
        trainer.test(model, test_loader)
