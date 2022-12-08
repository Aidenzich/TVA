#%%
from src.configs import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)

from src.datasets.common import RecsysData
from .utils import get_data
import pandas as pd

import inquirer


def handle_dataframe(df, config):
    # Rename columns for carre4 dataset
    df.rename(
        columns={
            config[USER_COLUMN_NAME]: USER_COLUMN_NAME,
            config[ITEM_COLUMN_NAME]: ITEM_COLUMN_NAME,
            config[RATING_COLUMN_NAME]: RATING_COLUMN_NAME,
            config[TIMESTAMP_COLUMN_NAME]: TIMESTAMP_COLUMN_NAME,
        },
        inplace=True,
    )

    dtypes = df.dtypes

    # filter value counts threshold
    # user_val_count = df[USER_COLUMN_NAME].value_counts()
    # user_index = user_val_count[user_val_count > config["count_threshold"]].index
    # remove_user_index = user_val_count[
    #     user_val_count <= config["count_threshold"]
    # ].index
    # removed_df = df[df[USER_COLUMN_NAME].isin(remove_user_index)]

    # df = df[df[USER_COLUMN_NAME].isin(user_index)]
    df[RATING_COLUMN_NAME] = df[RATING_COLUMN_NAME].astype(int)

    if (dtypes[TIMESTAMP_COLUMN_NAME] == "object") or (
        dtypes[TIMESTAMP_COLUMN_NAME] == "datetime64"
    ):
        df[TIMESTAMP_COLUMN_NAME] = (
            df[TIMESTAMP_COLUMN_NAME].astype("datetime64").astype("int64") // 10**9
        )

    print(df)
    return df


if __name__ == "__main__":
    data, data_path = get_data()

    if data == []:
        print("No data found in data folder")
        exit()

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

    df_cols = df.columns.tolist()
    questions = [
        inquirer.List(
            "user",
            message="Which column is the user column?",
            choices=df_cols,
        ),
        inquirer.List(
            "item",
            message="Which column is the item column?",
            choices=df_cols,
        ),
        inquirer.List(
            "rating",
            message="Which column is the rating column?",
            choices=df_cols,
        ),
        inquirer.List(
            "timestamp",
            message="Which column is the timestamp column?",
            choices=df_cols,
        ),
        inquirer.Text(
            "count_threshold",
            message="How many count threshold do you need?",
            default="3",
        ),
    ]

    answers = inquirer.prompt(questions)
    config = {
        USER_COLUMN_NAME: answers["user"],
        ITEM_COLUMN_NAME: answers["item"],
        RATING_COLUMN_NAME: answers["rating"],
        TIMESTAMP_COLUMN_NAME: answers["timestamp"],
        "count_threshold": int(answers["count_threshold"]),
    }
    df.dropna(
        subset=[
            config[USER_COLUMN_NAME],
            config[ITEM_COLUMN_NAME],
            config[RATING_COLUMN_NAME],
            config[TIMESTAMP_COLUMN_NAME],
        ],
        inplace=True,
    )

    df = handle_dataframe(df, config)
    newDataCLS = RecsysData(df=df, filename=choose_data_path.stem.lower())
    newDataCLS.save()
    print(f"Save dataclass into {newDataCLS._get_save_path()} Complete")


# %%
