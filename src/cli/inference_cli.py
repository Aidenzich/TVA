# %%
import pickle
import json
import inquirer

from .utils import get_dataclass, get_checkpoint_path
from src.configs import OUTPUT_PATH
from src.cli.utils import get_models
from src.models import INFER_FACTORY

if __name__ == "__main__":
    data_classes, data_classes_paths = get_dataclass()
    assert data_classes != [], "No dataclass found"

    models, model_paths = get_models()
    assert models != [], "No dataclass found"

    question = [
        inquirer.List(
            "model",
            message="Which model do you need?",
            choices=models,
        ),
        inquirer.List(
            "data_class",
            message="Which data class do you need? (Check in data/cache/dataclass)",
            choices=data_classes,
        ),
    ]

    answers = inquirer.prompt(question)
    dcls_path = data_classes_paths[data_classes.index(answers["data_class"])]
    model_path = model_paths[models.index(answers["model"])]

    # mymodel = VAECFModel.load_from_checkpoint()
    ckpts, ckpt_paths = get_checkpoint_path(model_path.name.lower())

    question = [
        inquirer.List(
            "checkpoint",
            message="Which model's checkpoint do you need?",
            choices=ckpts,
        ),
    ]
    answers = inquirer.prompt(question)
    ckpt_path = ckpt_paths[ckpts.index(answers["checkpoint"])]
    print(ckpt_path)

    print("Loading dataclass")

    with open(dcls_path, "rb") as f:
        recdata = pickle.load(f)

    print("Loading checkpoint")

    predict_result = INFER_FACTORY[model_path.name.lower()](ckpt_path, recdata, 10)
    predict_result = recdata.reverse_ids(recdata, predict_result)

    json.dump(
        predict_result,
        open(OUTPUT_PATH / f"{model_path.name.lower()}.{ckpt_path.name}.json", "w"),
        indent=2,
    )
