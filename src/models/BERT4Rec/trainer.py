from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.models.BERT4Rec.model import BERTModel
from src.abstrainer import ABSTrainer
from ray import tune
from src.config import LOG_PATH


class BERT4RecTrainer(ABSTrainer):
    def __init__(self, recdata, model_params, trainer_config):
        self.recdata = recdata
        self.model_params = model_params
        self.trainer_config = trainer_config

        if not self.trainer_config.get("tune"):
            self.train()
        else:
            tune_model_params = {}

            # Convert list to tune.choice for hyperparameter tuning
            for k, v in model_params.items():
                if type(v) == list:
                    tune_model_params[k] = tune.choice(v)
                else:
                    tune_model_params[k] = v

            self.model_params = tune_model_params
            self.tuner()

    def train(self):
        bert4rec_train(
            model_params=self.model_params,
            trainer_config=self.trainer_config,
            recdata=self.recdata,
        )

    def tuner(self):
        reporter = tune.CLIReporter(
            parameter_columns=["hidden_size", "n_layers", "dropout", "max_len", "head"],
            metric_columns=["recall"],
        )
        resources_per_trial = {"cpu": 12, "gpu": 1}
        train_fn_with_parameters = tune.with_parameters(
            bert4rec_train, recdata=self.recdata, trainer_config=self.trainer_config
        )

        analysis = tune.run(
            train_fn_with_parameters,
            resources_per_trial=resources_per_trial,
            progress_reporter=reporter,
            metric="recall",
            mode="max",
            config=self.model_params,
            local_dir=LOG_PATH / "tune",
            num_samples=1,  # trials number
            # scheduler=scheduler,
        )

        print("Best hyperparameters found were: ", analysis.best_config)


def bert4rec_train(model_params, trainer_config, recdata):
    # This can be store in the RecData class
    test_negative_sampler = NegativeSampler(
        train=recdata.train_seqs,
        val=recdata.val_seqs,
        test=recdata.test_seqs,
        user_count=recdata.num_users,
        item_count=recdata.num_items,
        sample_size=100,
        method="random",
        seed=12345,
    )

    test_negative_samples = test_negative_sampler.get_negative_samples()

    trainset = SequenceDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        seed=12345,
    )

    valset = SequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        negative_samples=test_negative_samples,
    )

    testset = SequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        negative_samples=test_negative_samples,
    )

    model = BERTModel(
        hidden_size=model_params["hidden_size"],
        num_items=recdata.num_items,
        n_layers=model_params["n_layers"],
        dropout=model_params["dropout"],
        heads=model_params["heads"],
        max_len=model_params["max_len"],
    )

    ABSTrainer.fit(
        model=model,
        trainset=trainset,
        valset=valset,
        trainer_config=trainer_config,
        model_params=model_params,
        testset=testset,
    )
