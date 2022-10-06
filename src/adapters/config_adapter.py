from src.config import DATA_PATH
from src.model import MODEL_FACTORY
import pickle
import json


def config_adapter(params_config, model_name):
    print("\033[93m" + json.dumps(params_config, sort_keys=True, indent=4) + "\033[0m")
    with open(DATA_PATH / "data_cls.pkl", "rb") as f:
        recsys_data = pickle.load(f)

    MODEL_FACTORY[model_name.lower()](recsys_data, params_config)
