from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.model.BERT4Rec.model import BERTModel
from src.config import DATA_PATH
from src.abstrain import ABSTrain
from torch.utils.data import DataLoader


class BERT4RecTrain(ABSTrain):
    def __init__(self, RecData, params_config):
        test_negative_sampler = NegativeSampler(
            train=RecData.train_seqs,
            val=RecData.val_seqs,
            test=RecData.test_seqs,
            user_count=RecData.num_users,
            item_count=RecData.num_items,
            sample_size=params_config["sample_size"],
            method="random",
            seed=12345,
        )

        test_negative_samples = test_negative_sampler.get_negative_samples()

        trainset = SequenceDataset(
            mode="train",
            max_len=params_config["max_len"],
            mask_prob=params_config["mask_prob"],
            num_items=RecData.num_items,
            mask_token=RecData.num_items + 1,
            u2seq=RecData.train_seqs,
            seed=12345,
        )

        valset = SequenceDataset(
            mode="eval",
            max_len=params_config["max_len"],
            mask_token=RecData.num_items + 1,
            u2seq=RecData.train_seqs,
            u2answer=RecData.val_seqs,
            negative_samples=test_negative_samples,
        )

        testset = SequenceDataset(
            mode="eval",
            max_len=params_config["max_len"],
            mask_token=RecData.num_items + 1,
            u2seq=RecData.train_seqs,
            u2answer=RecData.test_seqs,
            negative_samples=test_negative_samples,
        )

        test_loader = DataLoader(
            testset,
            batch_size=params_config["batch_size"],
            shuffle=False,
            pin_memory=True,
        )

        model = BERTModel(
            hidden_size=params_config["hidden_size"],
            num_items=RecData.num_items,  # item 的數量
            n_layers=params_config["n_layers"],
            dropout=params_config["dropout"],
            heads=params_config["heads"],
            max_len=params_config["max_len"],
        )

        self.fit(
            model=model,
            trainset=trainset,
            valset=valset,
            config=params_config,
            testset=testset,
        )
