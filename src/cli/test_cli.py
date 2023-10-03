import json
import yaml
import pickle
import inquirer

from .utils import get_models, get_configs, get_dataclass
from src.configs import DATACLASS_PATH, CONFIG_PATH

# from src.models import Trainer

import pytorch_lightning as pl
import warnings

# TODO
# if __name__ == "__main__":
#     models, models_path = get_models()
#     data_classes, data_classes_paths = get_dataclass()
#     questions = [
#         inquirer.List(
#             "data_class",
#             message="Which data class do you need? (Check in data/cache/dataclass)",
#             choices=data_classes,
#         ),
#         inquirer.List(
#             "model",
#             message="Which model do you need?",
#             choices=models,
#         ),
#     ]
#     answers = inquirer.prompt(questions)
#     midx = models.index(answers["model"])
#     dcls = answers["data_class"]
#     model_path = models_path[midx]
#     pl.Trainer()
#     trainer.test(model, dataloaders=DataLoader(test_set))
