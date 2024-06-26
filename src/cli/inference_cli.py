# %%
import pickle
import json
import inquirer

from .utils import get_dataclass, get_checkpoint_path, get_negative_samples
from src.configs import OUTPUT_PATH, RED_COLOR, END_COLOR
from src.cli.utils import get_models
from src.models import INFER_FACTORY
import signal


# %%
def keyboard_interrupt_handler(signal, frame) -> None:
    # Handle the keyboard interrupt here
    print(signal, frame)


def run_with_keyboard_interrupt_handler(func, *args, **kwargs):
    def wrapper(*args, **kwargs) -> None:
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            keyboard_interrupt_handler()

    return wrapper


@run_with_keyboard_interrupt_handler
def main() -> None:
    data_classes, data_classes_paths = get_dataclass()
    models, model_paths = get_models()
    nsamples, nsamples_paths = get_negative_samples()

    assert data_classes != [], RED_COLOR + "No dataclass found" + END_COLOR
    assert models != [], RED_COLOR + "No dataclass found" + END_COLOR
    assert data_classes != [], RED_COLOR + "No negative samples found" + END_COLOR

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
    model = answers["model"]
    model_path = model_paths[models.index(model)]

    ckpts, ckpt_paths = get_checkpoint_path(
        model_path.name.lower(), dcls_path.stem.lower()
    )

    print(ckpt_paths)
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

    if model.lower() in ["bert4rec", "tva2", "autoformer4rec"]:
        question = [
            inquirer.List(
                "nsample",
                message="Which negative samples do you need?",
                choices=[None] + nsamples,
            ),
        ]
        answers = inquirer.prompt(question)
        if answers["nsample"] != None:
            nsample_path = nsamples_paths[nsamples.index(answers["nsample"])]
            with open(nsample_path, "rb") as f:
                nsample = pickle.load(f)

        # Get the infer function from the selected model
        predict_result = INFER_FACTORY[model_path.name.lower()](
            ckpt_path=ckpt_path,
            recdata=recdata,
            rec_ks=top_k,
            negative_samples=nsample,
        )
    else:
        predict_result = INFER_FACTORY[model_path.name.lower()](
            ckpt_path=ckpt_path,
            recdata=recdata,
            rec_ks=top_k,
        )

    if predict_result != False:
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


if __name__ == "__main__":
    main()
