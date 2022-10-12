import json
import pickle
import inquirer

from .utils import get_models, get_configs
from src.configs import DATACLASS_PATH, CONFIG_PATH
from src.models.trainer import LightningTrainer


def create_configs_from_template(model_path, tune=False, config_name="default"):
    template_path = model_path / (
        model_path.name.lower() + (".tune" if tune else "") + ".template.config.json"
    )

    print(f"Creating configs from template {template_path}")
    with open(template_path, "r") as f:
        content = f.read()

    new_config_path = CONFIG_PATH / (
        model_path.name.lower()
        + f".{config_name}"
        + (".tune" if tune else "")
        + ".config.json"
    )

    with open(new_config_path, "w+") as f:
        f.write(content)

    print(f"Created {new_config_path} successfully")
    return new_config_path


def create_new_config_inquirer():
    questions = [
        inquirer.List(
            "istune",
            message="Do you want to auto-tune your model?",
            choices=["yes", "no"],
        ),
        inquirer.Text("config_name", message="What's config name? (blank: default)"),
    ]
    answers = inquirer.prompt(questions)

    tune = False
    if answers["istune"] == "yes":
        tune = True

    config_name = answers["config_name"] if answers["config_name"] else "default"
    selected_config_path = create_configs_from_template(
        model_path, tune=tune, config_name=config_name
    )


if __name__ == "__main__":
    models, models_path = get_models()
    which_models = [
        inquirer.List(
            "model",
            message="Which model do you need?",
            choices=models,
        ),
    ]

    answers = inquirer.prompt(which_models)

    midx = models.index(answers["model"])
    print(midx)
    model_path = models_path[midx]
    configs = get_configs(model_path.name)

    if configs == []:
        print(f'No config of {answers["model"]} found')
        create_new_config_inquirer()
    else:
        configs.append("➕ create new config")

        which_configs = [
            inquirer.List(
                "config",
                message="Which config do you need?",
                choices=configs,
            ),
        ]

        answers = inquirer.prompt(which_configs)

        if answers["config"] == "➕ create new config":
            create_new_config_inquirer()
        else:
            print("Using existing config")
            selected_config_path = answers["config"]

            config = json.load(open(selected_config_path))
            config["trainer_config"]["config_name"] = selected_config_path.name.replace(
                ".json", ""
            )

            with open(DATACLASS_PATH / config["data_class"], "rb") as f:
                recsys_data = pickle.load(f)

            LightningTrainer(
                recdata=recsys_data,
                model_params=config["model_params"],
                trainer_config=config["trainer_config"],
                model_name=model_path.name.lower(),
            )
