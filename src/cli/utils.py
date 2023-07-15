from pathlib import Path
from src.configs.paths import DATACLASS_PATH, LOG_PATH, DATA_PATH, NEGATIVE_SAMPLE_PATH

AVAILABLE_EXTENSIONS = ["csv", "pickle", "pkl"]


def get_models():
    p = Path(r"./src/models/").iterdir()
    model_paths = sorted([x for x in p if x.is_dir() and "__" not in str(x)])
    models = [x.name for x in model_paths]

    return models, model_paths


def get_configs(model_name, data_class="", extention="yaml"):
    p = Path(f"./configs").glob("**/*")

    configs = [
        x
        for x in p
        if x.is_file()
        and (model_name.lower() in str(x.name).split(".") and f".{extention}" in str(x))
        and (data_class.replace(".pkl", "").lower()) in str(x.name).split(".")
    ]

    configs = sorted(configs)

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


def get_negative_samples():
    p = Path(NEGATIVE_SAMPLE_PATH).glob("*.pkl")
    nsamples_path = [x for x in p if x.is_file()]
    nsamples = [x.name for x in nsamples_path]
    return nsamples, nsamples_path


def get_checkpoint_path(model_lower_name, data_class_stem):
    print(data_class_stem, model_lower_name)
    find_path = f"*{data_class_stem}/{model_lower_name}*/**/*.ckpt"
    print(find_path)
    p = Path(LOG_PATH).glob(find_path)
    ckpt_paths = [x for x in p if x.is_file()]
    ckpts = [str(x).replace(str(LOG_PATH), "")[1:] for x in ckpt_paths]
    return ckpts, ckpt_paths
