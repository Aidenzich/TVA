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
def plot_pca(data_matrix, transpose_matrix=True):
    """Use PCA to reduce dimension"""

    import matplotlib.pyplot as plt

    from sklearn.decomposition import PCA

    # use pca to reduce dimension for each row of dataframe
    pca = PCA(n_components=2)

    if transpose_matrix:
        data_matrix = pca.fit_transform(data_matrix.T.toarray())
    else:
        data_matrix = pca.fit_transform(data_matrix.toarray())

    x = data_matrix[:, 0]
    y = data_matrix[:, 1]
    # colors = plt.cm.coolwarm((x + y) / 2)

    plt.figure(figsize=(3, 3))
    plt.scatter(x, y, alpha=0.3)
    plt.show()


plot_pca(beauty.matrix, False)
plot_pca(toys.matrix, False)


# %%
def plot_svd(data_matrix, transpose_matrix=True):
    """Use SVD to reduce dimension"""

    import matplotlib.pyplot as plt

    from sklearn.decomposition import TruncatedSVD

    # use svd to reduce dimension for each row of dataframe
    svd = TruncatedSVD(n_components=2)

    if transpose_matrix:
        data_matrix = svd.fit_transform(data_matrix.T.toarray())
    else:
        data_matrix = svd.fit_transform(data_matrix.toarray())

    x = data_matrix[:, 0]
    y = data_matrix[:, 1]
    # colors = plt.cm.coolwarm((x + y) / 2)

    plt.figure(figsize=(3, 3))
    plt.scatter(x, y, alpha=0.3)
    plt.show()


plot_svd(beauty.matrix, False)
plot_svd(toys.matrix, False)

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
print(beauty_ucount)
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
# beauty = "beauty.pkl"
# with open(DATACLASS_PATH / beauty, "rb") as f:
#     beauty = pickle.load(f)

# with open(DATACLASS_PATH / "beauty_stable.pkl", "rb") as f:
#     beauty_stable = pickle.load(f)

# for u in tqdm(beauty.users_seqs.keys()):
#     if beauty.users_seqs[u] == beauty_stable.users_seqs[u]:
#         print("true")

# %%
