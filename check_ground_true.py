#%%
import pickle
from src.config import DATA_PATH

with open(DATA_PATH / "data_cls.pkl", "rb") as f:
    recsys_data = pickle.load(f)

# %%
recsys_data.dataframe
# %%
top10popular = recsys_data.dataframe.item_id.value_counts()[:10].index.tolist()


#%%
from tqdm import tqdm

hits = 0
for i in tqdm(recsys_data.test_seqs):
    if i in top10popular:
        hits += 1

print(round(hits / len(recsys_data.test_seqs), 4))

# %%
