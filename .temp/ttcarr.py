#%%
import sys

sys.path.append("../")
sys.path.append("../../")
import sys

from matplotlib.style import available
from regex import D

from src.configs.paths import DATA_PATH, ROOT_PATH
from src.configs import (
    USER_COLUMN_NAME,
    ITEM_COLUMN_NAME,
    RATING_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)

sys.path.append("../")
sys.path.append("../../")

import json
import pickle
import inquirer
from pathlib import Path
import pandas as pd
from src.configs import DATACLASS_PATH, CONFIG_PATH

df = pd.read_pickle(DATA_PATH / "carrefour_all.pkl")

config = {
    USER_COLUMN_NAME: "customer",
    ITEM_COLUMN_NAME: "product",
    RATING_COLUMN_NAME: "quantity",
    TIMESTAMP_COLUMN_NAME: "order_date",
    "least_threshold": 10,
}

# df = pd.read_csv(data_path)

df.rename(
    columns={
        config[USER_COLUMN_NAME]: USER_COLUMN_NAME,
        config[ITEM_COLUMN_NAME]: ITEM_COLUMN_NAME,
        config[RATING_COLUMN_NAME]: RATING_COLUMN_NAME,
    },
    inplace=True,
)

dtypes = df.dtypes

user_val_count = df.user_id.value_counts()
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

#%%
df