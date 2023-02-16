#%%
import sys

sys.path.append("../../")
import pickle
from src.configs import DATACLASS_PATH, DATA_PATH, NEGATIVE_SAMPLE_PATH
from src.datasets.negative_sampler import NegativeSampler
from tqdm import tqdm

data_class_name = "movielens_cls.pkl"
print(DATACLASS_PATH)
with open(DATACLASS_PATH / data_class_name, "rb") as f:
    recsys_data = pickle.load(f)

#%%
recsys_data.matrix[:, 0].A[0]

#%%
import numpy as np

np.array(recsys_data.matrix.A[:, 0]).shape

# %%
from src.datasets.matrix_dset import MatrixDataset

# %%
myset = MatrixDataset(recsys_data.train_matrix, wise="column")

# %%
myset[1].shape
# %%
