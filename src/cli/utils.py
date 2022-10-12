from pathlib import Path
from src.configs.paths import DATACLASS_PATH, LOG_PATH, DATA_PATH

AVAILABLE_EXTENSIONS = ["csv", "pickle", "pkl"]


def get_models():
    p = Path(r"./src/models/").iterdir()
    model_paths = [x for x in p if x.is_dir() and "__" not in str(x)]
    models = [x.name for x in model_paths]
    return models, model_paths


def get_configs(model_name):
    p = Path(r"./configs").glob("**/*")
    configs = [
        x
        for x in p
        if x.is_file() and (model_name.lower() in str(x) and "config.json" in str(x))
    ]
    return configs


def get_data():
    p = Path(DATA_PATH).glob("*")
    data_paths = [
        x
        for x in p
        if (x.is_file())
        and (not x.is_dir())
        and (x.name.split(".")[-1] in AVAILABLE_EXTENSIONS)
    ]
    data = [x.name for x in data_paths]

    return data, data_paths


def get_dataclass():
    p = Path(DATACLASS_PATH).glob("*.pkl")
    data_classes_path = [x for x in p if x.is_file()]
    data_classes = [x.name for x in data_classes_path]
    return data_classes, data_classes_path


def get_checkpoint_path(model_lower_name):
    p = Path(LOG_PATH).glob(f"*{model_lower_name}*/**/*.ckpt")
    ckpt_paths = [x for x in p if x.is_file()]
    ckpts = [str(x).replace(str(LOG_PATH), "")[1:] for x in ckpt_paths]
    return ckpts, ckpt_paths
