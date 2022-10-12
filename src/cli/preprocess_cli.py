#%%
import sys

from matplotlib.style import available
from regex import D

from src.configs.paths import DATA_PATH, ROOT_PATH, DATACLASS_PATH, CONFIG_PATH
from src.configs import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)

from src.datasets.common import RecsysData
import pandas as pd

import inquirer
from pathlib import Path

AVAILABLE_EXTENSIONS = ["csv", "pickle", "pkl"]


def get_data():
    p = Path(DATA_PATH).glob("*")
    p2 = Path(DATA_PATH).glob("*")  # Python can't clone a generator
    data = [
        x.name
        for x in p
        if (x.is_file())
        and (not x.is_dir())
        and (x.name.split(".")[-1] in AVAILABLE_EXTENSIONS)
    ]

    data_path = [
        x
        for x in p2
        if (x.is_file())
        and (not x.is_dir())
        and (x.name.split(".")[-1] in AVAILABLE_EXTENSIONS)
    ]
    return data, data_path


def handle_dataframe(df, config):

    df.rename(
        columns={
            config[USER_COLUMN_NAME]: USER_COLUMN_NAME,
            config[ITEM_COLUMN_NAME]: ITEM_COLUMN_NAME,
            config[RATING_COLUMN_NAME]: RATING_COLUMN_NAME,
        },
        inplace=True,
    )

    dtypes = df.dtypes

    # filter value counts threshold
    user_val_count = df[USER_COLUMN_NAME].value_counts()
    user_index = user_val_count[user_val_count > config["least_threshold"]].index
    df = df[df[USER_COLUMN_NAME].isin(user_index)]

    if dtypes[config[TIMESTAMP_COLUMN_NAME]] == "object":
        df[TIMESTAMP_COLUMN_NAME] = (
            df[config[TIMESTAMP_COLUMN_NAME]].astype("datetime64").astype("int64")
            // 10**9
        )

    if dtypes[config[TIMESTAMP_COLUMN_NAME]] == "datetime64":
        df[TIMESTAMP_COLUMN_NAME] = (
            df[config[TIMESTAMP_COLUMN_NAME]].astype("int64") // 10**9
        )

    return df


if __name__ == "__main__":
    data, data_path = get_data()

    if data == []:
        print("No data found in data folder")
        exit()

    # data = [1, 2]
    questions = [
        inquirer.List("data", message="Which data do you want to use?", choices=data)
    ]

    answers = inquirer.prompt(questions)
    choose_data = answers["data"]

    choose_data_path = data_path[data.index(choose_data)]
    choose_data_extension = choose_data.split(".")[-1]

    if choose_data_extension == "csv":
        df = pd.read_csv(choose_data_path)
    elif choose_data_extension == "pkl" or choose_data_extension == "pickle":
        df = pd.read_pickle(choose_data_path)
    config = {
        USER_COLUMN_NAME: "customer",
        ITEM_COLUMN_NAME: "product",
        RATING_COLUMN_NAME: "quantity",
        TIMESTAMP_COLUMN_NAME: "order_date",
        "least_threshold": 10,
    }

    df = handle_dataframe(df, config)
    newDataCLS = RecsysData(df=df, filename=choose_data_path.stem)
    newDataCLS.save()
    print(f"Save dataclass into {newDataCLS._get_save_path()} Complete")


# %%
