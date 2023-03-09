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
recsys_data.users_seqs
recsys_data.users_timeseqs

# %%
import datetime

x = {}

for u in tqdm(recsys_data.users_timeseqs):
    dates = [datetime.datetime.fromtimestamp(t) for t in recsys_data.users_timeseqs[u]]
    years = [d.year if d is not None else 0 for d in dates]
    months = [d.month if d is not None else 0 for d in dates]
    days = [d.day if d is not None else 0 for d in dates]
    hours = [d.hour if d is not None else 0 for d in dates]
    minutes = [d.minute if d is not None else 0 for d in dates]
    seconds = [d.second if d is not None else 0 for d in dates]

    items = recsys_data.users_seqs[u]

    for idx, item in enumerate(items):

        if item not in x:
            x[item] = {
                "years": [],
                "months": [],
                "days": [],
                "hours": [],
                "minutes": [],
                "seconds": [],
            }

        x[item]["years"].append(years[idx])
        x[item]["months"].append(months[idx])
        x[item]["days"].append(days[idx])
        x[item]["hours"].append(hours[idx])
        x[item]["minutes"].append(minutes[idx])
        x[item]["seconds"].append(seconds[idx])


# %%
# list(x.keys())[0]
max_years = 0
max_months = 0
min_years = 9999
min_months = 0

for i in x:
    years_list = x[i]["years"]
    max_years = max(max_years, max(years_list))
    min_years = min(min_years, min(years_list))

x2 = {}

for i in x:
    year_range = (max_years - min_years + 1)
    year_count = [0] * year_range
    
    for i in x2[i]["years"]:
        year_count[i - min_years] += 1
    
    x2[i]["years"][]


# %%
print(min_years, max_years)

# %%
