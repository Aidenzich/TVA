# %%
import sys

sys.path.append("../../")

from utils import generate_seqlen_group
from src.models.TVA4.model import TVAModel
from src.datasets.tva_dset import TVASequenceDataset
from src.configs import DATACLASS_PATH
import numpy as np
import pickle
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

seed_everything(0, workers=True)
user_latent_factor = None

# # Beauty

beauty = "beauty.pkl"

# Toys
toys = "toys.pkl"

# ML1m
ml1m = "ml1m.pkl"


with open(DATACLASS_PATH / beauty, "rb") as f:
    beauty = pickle.load(f)

with open(DATACLASS_PATH / toys, "rb") as f:
    toys = pickle.load(f)

with open(DATACLASS_PATH / ml1m, "rb") as f:
    ml1m = pickle.load(f)


# %%
beauty_icount = beauty.dataframe.item_id.value_counts()
toys_icount = toys.dataframe.item_id.value_counts()

print("== beauty ==")
print(beauty_icount.describe())
print("== toys ==")
print(toys_icount.describe())
print(toys_icount)

# plot count data
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.plot(beauty_icount.values, label="beauty")
plt.plot(toys_icount.values, label="toys")
plt.legend()
plt.show()


# %%
beauty_ucount = beauty.dataframe.user_id.value_counts()
toys_ucount = toys.dataframe.user_id.value_counts()
print("== beauty ==")
print(beauty_ucount.describe())
print("== toys ==")
print(toys_ucount.describe())
print(toys_ucount)
# plot count data
plt.figure(figsize=(5, 5))
plt.plot(beauty_ucount.values, label="beauty")
plt.plot(toys_ucount.values, label="toys")
plt.legend()
plt.show()

# %%
beauty = "beauty.pkl"
with open(DATACLASS_PATH / beauty, "rb") as f:
    beauty = pickle.load(f)

with open(DATACLASS_PATH / "beauty_stable.pkl", "rb") as f:
    beauty_stable = pickle.load(f)

# %%
for u in tqdm(beauty.users_seqs.keys()):
    if beauty.users_seqs[u] == beauty_stable.users_seqs[u]:
        print("true")
