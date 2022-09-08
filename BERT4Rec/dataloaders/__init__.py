from tmpdatasets import dataset_factory
from .bert import BertDataloader


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = BertDataloader
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
