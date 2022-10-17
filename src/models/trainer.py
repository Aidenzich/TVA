from abc import ABCMeta
from src.models import TRAIN_FACTORY
from src.adapters.raytune_adapter import tuner


class LightningTrainer(metaclass=ABCMeta):
    def __init__(self, recdata, model_params, trainer_config, model_name):
        self.model_name = model_name
        self.recdata = recdata
        self.model_params = model_params
        self.trainer_config = trainer_config

        if not self.trainer_config.get("tune"):
            self.train()
        else:
            self.tune()

    def print_model_params(self):
        print(self.model_params)

    def print_trainer_config(self):
        print(self.trainer_config)

    def train(self):

        TRAIN_FACTORY[self.model_name](
            model_params=self.model_params,
            trainer_config=self.trainer_config,
            recdata=self.recdata,
        )

    def tune(self):
        self.recdata.dataframe = None
        tuner(
            model_params=self.model_params,
            trainer_config=self.trainer_config,
            dataclass=self.recdata,
            model_name=self.model_name,
        )
