import json
import yaml
import pickle
import inquirer

from .utils import get_models, get_configs, get_dataclass
from src.configs import DATACLASS_PATH, CONFIG_PATH
from src.models import Trainer
import warnings

warnings.filterwarnings("ignore", ".*(log_every_n_steps).*")


def create_configs_from_template(
    model_path, data_class_name, data_class_stem, tune=False, config_name="default"
):
    template_path = model_path / (("tune." if tune else "") + "template.json")

    print(f"Creating configs from template {template_path}")

    config = json.load(open(template_path, "r"))
    config["data_class"] = data_class_name

    yaml.dump(config, open(template_path.with_suffix(".yaml"), "w"))

    new_config_path = CONFIG_PATH / (
        f"{data_class_stem}"  # data class name
        + f".{model_path.name.lower()}"  # model name
        + f".{config_name}"
        + (".tune" if tune else "")  # tune or not
        + ".json"
    )

    json.dump(config, open(new_config_path, "w"), indent=2)

    print(f"Created {new_config_path} successfully")
    return new_config_path


def create_new_config_inquirer() -> None:
    data_classes, data_classes_paths = get_dataclass()
    assert data_classes != [], "No dataclass found"

    questions = [
        inquirer.List(
            "istune",
            message="Do you want to auto-tune your model?",
            choices=["yes", "no"],
        ),
        inquirer.List(
            "data_class",
            message="Which data class do you need? (Check in data/cache/dataclass)",
            choices=data_classes,
        ),
        inquirer.Text("config_name", message="What's config name? (blank: default)"),
    ]
    answers = inquirer.prompt(questions)
    dcls_path = data_classes_paths[data_classes.index(answers["data_class"])]

    tune = False
    if answers["istune"] == "yes":
        tune = True

    config_name = answers["config_name"] if answers["config_name"] else "default"
    selected_config_path = create_configs_from_template(
        model_path,
        tune=tune,
        config_name=config_name,
        data_class_name=dcls_path.name.lower(),
        data_class_stem=dcls_path.stem.lower(),
    )


if __name__ == "__main__":
    models, models_path = get_models()
    data_classes, data_classes_paths = get_dataclass()
    questions = [
        inquirer.List(
            "data_class",
            message="Which data class do you need? (Check in data/cache/dataclass)",
            choices=data_classes,
        ),
        inquirer.List(
            "model",
            message="Which model do you need?",
            choices=models,
        ),
    ]

    answers = inquirer.prompt(questions)
    midx = models.index(answers["model"])

    dcls = answers["data_class"]
    
    model_path = models_path[midx]
    configs = get_configs(model_name=model_path.name, data_class=dcls)
    
    
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

            # config = json.load(open(selected_config_path))
            config = yaml.safe_load(open(selected_config_path))

            config["trainer_config"]["config_name"] = selected_config_path.name.replace(
                ".json", ""
            ).replace(".yaml", "")

            # yaml.dump(config, open(selected_config_path.with_suffix(".yaml"), "w"))

            with open(DATACLASS_PATH / config["data_class"], "rb") as f:
                recsys_data = pickle.load(f)

            recsys_data.show_info_table()

            Trainer(
                recdata=recsys_data,
                model_params=config["model_params"],
                trainer_config=config["trainer_config"],
                model_name=model_path.name.lower(),
            )
