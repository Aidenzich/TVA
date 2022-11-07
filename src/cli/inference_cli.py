# %%
import pickle
import json
import inquirer

from .utils import get_dataclass, get_checkpoint_path
from src.configs import OUTPUT_PATH, RED_COLOR, END_COLOR
from src.cli.utils import get_models
from src.models import INFER_FACTORY

if __name__ == "__main__":
    data_classes, data_classes_paths = get_dataclass()
    assert data_classes != [], RED_COLOR + "No dataclass found" + END_COLOR

    models, model_paths = get_models()
    assert models != [], RED_COLOR + "No dataclass found" + END_COLOR

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
        inquirer.Text(
            "top_k",
            message="How many top k do you need?",
            default="5",
        ),
    ]

    answers = inquirer.prompt(question)

    assert answers["top_k"].isdigit(), RED_COLOR + "Top k must be a number" + END_COLOR
    top_k = int(answers["top_k"])

    dcls_path = data_classes_paths[data_classes.index(answers["data_class"])]
    model_path = model_paths[models.index(answers["model"])]

    ckpts, ckpt_paths = get_checkpoint_path(
        model_path.name.lower(), dcls_path.stem.lower()
    )

    assert ckpt_paths != [], RED_COLOR + "No checkpoint found" + END_COLOR
    question = [
        inquirer.List(
            "checkpoint",
            message="Which model's checkpoint do you need?",
            choices=ckpts,
        ),
    ]
    answers = inquirer.prompt(question)
    ckpt_path = ckpt_paths[ckpts.index(answers["checkpoint"])]

    print("Loading dataclass")

    with open(dcls_path, "rb") as f:
        recdata = pickle.load(f)

    print("Loading checkpoint")

    predict_result = INFER_FACTORY[model_path.name.lower()](
        ckpt_path=ckpt_path, recdata=recdata, rec_ks=top_k
    )
    predict_result = recdata.reverse_ids(recdata, predict_result)

    json.dump(
        predict_result,
        open(
            OUTPUT_PATH
            / f"{ckpt_path.parent.parent.parent.name}.{ckpt_path.stem.lower()}.json",
            "w",
        ),
        indent=2,
    )

    print("Infer Complete")
