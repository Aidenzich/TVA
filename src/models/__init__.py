from .BERT4Rec import train_bert4rec, infer_bert4rec
from .VAECF import train_vaecf, infer_vaecf
from .TVA2 import train_tva2, infer_tva2
from .TVA3 import train_tva2 as train_tva3
from .Autoformer4Rec import train as train_auto

TRAIN_FACTORY = {
    "bert4rec": train_bert4rec,
    "vaecf": train_vaecf,
    "tva2": train_tva2,
    "tva3": train_tva3,
    "autoformer4rec": train_auto,
}


INFER_FACTORY = {"bert4rec": infer_bert4rec, "vaecf": infer_vaecf, "tva": infer_tva2}


from abc import ABCMeta
from src.models import TRAIN_FACTORY
from src.adapters.raytune_adapter import tuner
from pytorch_lightning.utilities.seed import seed_everything


class Trainer(metaclass=ABCMeta):
    def __init__(self, recdata, model_params, trainer_config, model_name) -> None:
        self.model_name = model_name
        self.recdata = recdata
        self.model_params = model_params
        self.trainer_config = trainer_config
        seed_everything(trainer_config["seed"])
        if not self.trainer_config.get("tune"):
            self.train()
        else:
            self.tune()

    def print_model_params(self) -> None:
        print(self.model_params)

    def print_trainer_config(self) -> None:
        print(self.trainer_config)

    def train(self) -> None:

        TRAIN_FACTORY[self.model_name](
            model_params=self.model_params,
            trainer_config=self.trainer_config,
            recdata=self.recdata,
        )

    def tune(self) -> None:
        self.recdata.dataframe = None
        tuner(
            model_params=self.model_params,
            trainer_config=self.trainer_config,
            dataclass=self.recdata,
            model_name=self.model_name,
        )
