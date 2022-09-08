import torch
from options import args
from model import BERTModel
from dataloaders import dataloader_factory
from utils import *
import pytorch_lightning as pl
from templates import set_template

def train():
    set_template(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)    
    model = BERTModel(args)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=10, gpus=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
