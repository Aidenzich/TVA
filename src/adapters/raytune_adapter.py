from ray import tune, air
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from src.configs import LOG_PATH
from src.models import TRAIN_FACTORY
from typing import Dict, Any, Tuple


def tuner(model_params, trainer_config, dataclass, model_name) -> None:
    __list2tunechoice__(model_params)
    tune_config = trainer_config["tune"]["tune_config"]
    resources_per_trial = trainer_config["tune"]["resources_per_trial"]
    dataclass.dataframe = None
    config = __get_tuneconfig__(model_params, trainer_config, dataclass, model_name)
    analysis = tune.Tuner(
        tune.with_resources(__trainable_function__, resources_per_trial),
        tune_config=tune.TuneConfig(
            metric=tune_config["metric"],
            mode=tune_config["mode"],
            num_samples=tune_config["num_samples"],
        ),
        param_space=config,
        run_config=air.RunConfig(
            name=trainer_config["config_name"], local_dir=LOG_PATH
        ),
    ).fit()

    print("Best hyperparameters found were: ", analysis.get_best_result().config)


# Ray train only need one parameter: the config
def __trainable_function__(config: dict) -> None:
    model_params, trainer_config, dataclass = __rollback_tuneconfig__(config)
    tune_config = trainer_config.get("tune")

    callbacks = [TuneReportCallback(tune_config["metrics"], on="validation_end")]
    TRAIN_FACTORY[config["model_name"]](
        model_params=model_params,
        trainer_config=trainer_config,
        recdata=dataclass,
        callbacks=callbacks,
    )


def __get_tuneconfig__(
    model_params: dict,
    trainer_config: dict,
    dataclass,
    model_name: str,
) -> Dict[str, Any]:
    config = {
        "model_name": model_name,
        "model_params_columns": list(model_params.keys()),
        "trainer_config_columns": list(trainer_config.keys()),
        "dataclass": dataclass,
        **model_params,
        **trainer_config,
    }
    return config


def __rollback_tuneconfig__(config: dict) -> Tuple[dict, dict, Any]:
    model_params = {k: config[k] for k in config["model_params_columns"]}
    trainer_config = {k: config[k] for k in config["trainer_config_columns"]}
    dataclass = config["dataclass"]
    return model_params, trainer_config, dataclass


def __list2tunechoice__(config: dict) -> dict:
    for k, v in config.items():
        if type(v) == list:
            config[k] = tune.choice(v)
    return config
