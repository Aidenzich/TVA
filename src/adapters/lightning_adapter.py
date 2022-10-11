import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.configs import LOG_PATH


def fit(
    model,
    trainset,
    trainer_config,
    model_params,
    valset=None,
    testset=None,
    callbacks=[],
):

    early_stop_config = trainer_config.get("early_stopping")
    if trainer_config.get("early_stopping"):
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get("monitor"),
            patience=early_stop_config.get("patience"),
            verbose=early_stop_config.get("verbose"),
            mode=early_stop_config.get("mode"),
        )
        callbacks.append(early_stop_callback)

    train_loader = DataLoader(
        trainset,
        batch_size=model_params["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    if valset != None:
        val_loader = DataLoader(
            valset,
            batch_size=model_params["batch_size"],
            shuffle=False,
            pin_memory=True,
        )
    else:
        val_loader = None

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=LOG_PATH)

    trainer = pl.Trainer(
        callbacks=callbacks,
        limit_train_batches=trainer_config["limit_train_batches"],
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
