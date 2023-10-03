# %%
from cornac.datasets import (
    movielens,
    netflix,
)
import inquirer
import numpy as np
import pandas as pd
from src.datasets.base import RecsysData

# %%
datasets_options_factory = {
    "movielens": movielens,
    "netflix": netflix,
    # "amazon_clothing": amazon_clothing,
    # "amazon_digital_music": amazon_digital_music,
    # "amazon_office": amazon_office,
    # "amazon_toy": amazon_toy,
    # "citeulike": citeulike,
    # "epinions": epinions,
    # "filmtrust": filmtrust,
    # "tradesy": tradesy,
}


question = [
    inquirer.List(
        "datasets",
        message="Which datasets from cornac do you need?",
        choices=list(datasets_options_factory.keys()),
    )
]

answers = inquirer.prompt(question)

print(f'Loading {answers["datasets"]}...')

data = datasets_options_factory[answers["datasets"]].load_feedback(
    fmt="UIRT", variant="1M"
)
print(data[-1])


# data = datasets_options_factory["movielens"].load_feedback(fmt="UIRT")
data = np.array(data)

datarame = {
    "user_id": data[:, 0],
    "item_id": data[:, 1],
    "rating": data[:, 2],
    "timestamp": data[:, 3],
}

df = pd.DataFrame(datarame)
newDataCLS = RecsysData(df=df, filename="movielens")
newDataCLS.save()
print(f"Save dataclass into {newDataCLS._get_save_path()} Complete")
