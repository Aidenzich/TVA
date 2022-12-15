# %%
import pickle
import json
import inquirer

from .utils import get_dataclass, get_checkpoint_path, get_negative_samples
from src.configs import (
    OUTPUT_PATH,
    RED_COLOR,
    END_COLOR,
    ITEM_COLUMN_NAME,
    USER_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)
from src.cli.utils import get_models
from src.models import INFER_FACTORY

import numpy as np
from tqdm import tqdm

#%%
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

answers = {
    "model": "BERT4Rec",
    "data_class": "otto_test_cls.pkl",
    "top_k": 20,
    "checkpoint": "bert4rec.default.otto_test_cls.config/version_4/checkpoints/epoch=3-step=466828.ckpt",
    "nsample": "otto_test_nsample_for_carts.pkl"
}


assert answers["top_k"].isdigit(), RED_COLOR + "Top k must be a number" + END_COLOR

top_k = int(answers["top_k"])

dcls_path = data_classes_paths[data_classes.index(answers["data_class"])]
model_path = model_paths[models.index(answers["model"])]

ckpts, ckpt_paths = get_checkpoint_path(model_path.name.lower(), dcls_path.stem.lower())

assert ckpt_paths != [], RED_COLOR + "No checkpoint found" + END_COLOR



ckpt_path = ckpt_paths[ckpts.index(answers["checkpoint"])]

print("Loading dataclass")

with open(dcls_path, "rb") as f:
    recdata = pickle.load(f)

print("Loading checkpoint")


if answers["nsample"] != None:
    nsample_path = nsamples_paths[nsamples.index(answers["nsample"])]
    with open(nsample_path, "rb") as f:
        nsample = pickle.load(f)

users = recdata.dataframe[USER_COLUMN_NAME].unique()
user_group = recdata.dataframe.groupby(USER_COLUMN_NAME)
user2items = user_group.progress_apply(
    lambda d: list(d.sort_values(by=TIMESTAMP_COLUMN_NAME)[ITEM_COLUMN_NAME])
)
user2time = user_group.progress_apply(
    lambda t: list(
        t.sort_values(by=TIMESTAMP_COLUMN_NAME)[TIMESTAMP_COLUMN_NAME].astype(np.int64)
    )
)

for user in tqdm(users):
    items = user2items[user]
    timestamps = user2time[user]

    train_seqs, val_seqs, test_seqs, fully_seqs = {}, {}, {}, {}
    train_timeseqs, val_timeseqs, test_timeseqs, fully_timeseqs = {}, {}, {}, {}

    train_seqs[user], val_seqs[user], test_seqs[user], fully_seqs[user] = (
        items[:-2],
        items[-2:-1],
        items[-1:],
        items,
    )

    (
        train_timeseqs[user],
        val_timeseqs[user],
        test_timeseqs[user],
        fully_timeseqs[user],
    ) = (
        timestamps[:-2],
        timestamps[-2:-1],
        timestamps[-1:],
        timestamps,
    )


recdata.train_seqs = fully_seqs

# Get the infer function from the selected model
predict_result = INFER_FACTORY[model_path.name.lower()](
    ckpt_path=ckpt_path,
    recdata=recdata,
    rec_ks=top_k,
    negative_samples=nsample,
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
