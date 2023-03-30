import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from src.configs import LOG_PATH
from pytorch_lightning.callbacks import ModelCheckpoint


def fit(
    model,
    trainset,
    trainer_config,
    model_params,
    valset=None,
    testset=None,
    callbacks=[],
) -> None:
    """_summary_
    The function trains the model on the trainset using the provided trainer_config and model_params.
    It may also evaluate the trained model on the valset and testset if provided,
    and apply any callbacks during training.

    Args:
        model (_type_): _description_
        trainset (_type_): _description_
        trainer_config (_type_): _description_
        model_params (_type_): _description_
        valset (_type_, optional): _description_. Defaults to None.
        testset (_type_, optional): _description_. Defaults to None.
        callbacks (list, optional): _description_. Defaults to [].
    """

    early_stop_config = trainer_config.get("early_stopping")

    # early stopping
    if trainer_config.get("early_stopping"):
        print("Early stopping is enabled")
        early_stop_callback = EarlyStopping(
            monitor=early_stop_config.get("monitor"),
            patience=early_stop_config.get("patience"),
            verbose=early_stop_config.get("verbose"),
            mode=early_stop_config.get("mode"),
        )
        callbacks.append(early_stop_callback)

    # save top k checkpoints
    if trainer_config.get("save_checkpoint"):
        checkpoint_callback = ModelCheckpoint(
            dirpath=LOG_PATH,
            save_top_k=trainer_config["save_checkpoint"]["save_top_k"],
            monitor=trainer_config["save_checkpoint"]["monitor"],
        )
        callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    train_loader = DataLoader(
        trainset,
        batch_size=model_params["batch_size"],
        shuffle=True,
        pin_memory=False,
        num_workers=trainer_config["num_workers"],
    )
    if valset != None:
        val_loader = DataLoader(
            valset,
            batch_size=model_params["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=trainer_config["num_workers"],
        )
    else:
        val_loader = None

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=LOG_PATH, name=trainer_config["config_name"]
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        limit_train_batches=trainer_config["limit_train_batches"],
        max_epochs=trainer_config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=tb_logger,
        devices=trainer_config.get("devices", [0]),
        check_val_every_n_epoch=trainer_config.get("check_val_every_n_epoch"),
        # For reproducibility
        deterministic=True,
        gradient_clip_val=trainer_config.get("gradient_clip_val", 0),
    )
    trainer.fit(model, train_loader, val_loader)

    if testset != None:
        test_loader = DataLoader(
            testset,
            batch_size=model_params["batch_size"],
            shuffle=False,
            pin_memory=False,
            num_workers=trainer_config["num_workers"],
        )
        trainer.test(model, test_loader)
