import inquirer
from pathlib import Path
from ..config import CONFIG_PATH
from ..adapters import config_adapter
import json


def get_models():
    p = Path(r"./src/model/").iterdir()
    models = [x for x in p if x.is_dir()]
    return models


def get_configs(model_name):
    p = Path(r"./configs").glob("**/*")
    configs = [
        x for x in p if x.is_file() and (model_name.lower() and "config.json" in str(x))
    ]
    return configs


def create_configs_from_template(model_path, tune=False, config_name="default"):
    template_path = model_path / (
        model_path.name.lower() + (".tune" if tune else "") + ".config.template"
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
    models = get_models()
    which_models = [
        inquirer.List(
            "model",
            message="Which model do you need?",
            choices=models,
        ),
    ]

    answers = inquirer.prompt(which_models)
    model_path = answers["model"]
    configs = get_configs(model_path.name)

    if configs == []:
        print(f'No config of {answers["model"]} found')
        create_new_config_inquirer()
    else:
        configs.append("🪑 create new config")

        which_configs = [
            inquirer.List(
                "config",
                message="Which config do you need?",
                choices=configs,
            ),
        ]

        answers = inquirer.prompt(which_configs)
        print(answers)
        if answers["config"] == "🪑 create new config":
            create_new_config_inquirer()
        else:
            print("Using existing config")
            selected_config_path = answers["config"]

            config = json.load(open(selected_config_path))

            config_adapter(config["params"], model_name=model_path.name)
