from .TVA4 import train as train_tva4
from .TVA6 import train as train_tva6
from .TVAE import train as train_tvae


from .VAECF import train as train_vaecf, infer as infer_vaecf
from .SASRec import train as train_sas, infer as infer_sasrec
from .VAEICF import train as train_vaeicf, infer as infer_vaeicf
from .BERT4Rec import train as train_bert4rec, infer as infer_bert4rec
from .BERT4RecS import train as train_bert4recs, infer as infer_bert4recs
from .CBiT import train as train_cbit
from .ContrastVAE import train as train_cvae
from tabulate import tabulate

from ..configs import CYAN_COLOR, END_COLOR

TRAIN_FACTORY = {
    "bert4rec": train_bert4rec,
    "vaecf": train_vaecf,
    "vaeicf": train_vaeicf,
    "tva4": train_tva4,
    "tva6": train_tva6,
    "sasrec": train_sas,
    "tvae": train_tvae,
    "bert4recs": train_bert4recs,
    "cbit": train_cbit,
    "contrastvae": train_cvae,
}
INFER_FACTORY = {
    "bert4rec": infer_bert4rec,
    "vaecf": infer_vaecf,
    "vaeicf": infer_vaeicf,
    "sasrec": infer_sasrec,
}


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
        seed_everything(trainer_config["seed"], workers=True)
        if not self.trainer_config.get("tune"):
            self.train()
        else:
            self.tune()

    def print_model_params(self) -> None:
        data = []
        for k, v in self.model_params.items():
            if type(v) == dict:
                for k1, v1 in v.items():
                    data.append([k1, v1])
            else:
                data.append([k, v])

        print(
            CYAN_COLOR + "\n",
            "Model Parameters: \n",
            tabulate(
                data,
                headers=["property", "value"],
                tablefmt="heavy_outline",
                numalign="right",
            ),
            END_COLOR + "\n",
        )

    def print_trainer_config(self) -> None:

        data = []
        for k, v in self.trainer_config.items():
            if type(v) == dict:
                for k1, v1 in v.items():
                    data.append([k1, v1])
            else:
                data.append([k, v])

        print(
            CYAN_COLOR + "\n",
            "Trainer Configs: \n",
            tabulate(
                data,
                headers=["property", "value"],
                tablefmt="heavy_outline",
                numalign="right",
            ),
            END_COLOR + "\n",
        )

    def train(self) -> None:
        self.print_model_params()
        self.print_trainer_config()
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
