#%%
import json
import pickle
from src.configs import DATACLASS_PATH, DATA_PATH, NEGATIVE_SAMPLE_PATH
from src.datasets.negative_sampler import NegativeSampler
from tqdm import tqdm

data_class_name = "otto_test_cls.pkl"
print(DATACLASS_PATH)
with open(DATACLASS_PATH / data_class_name, "rb") as f:
    recsys_data = pickle.load(f)

with open(NEGATIVE_SAMPLE_PATH / "otto_test_nsample_for_carts.pkl", "rb") as f:
    nsample = pickle.load(f)


#%%
test_negative_sampler = NegativeSampler(
    train=recsys_data.train_seqs,
    val=recsys_data.val_seqs,
    test=recsys_data.test_seqs,
    item_count=recsys_data.num_items,
    sample_size=100,
    method="popular",
    seed=1250,
    dataclass_name=recsys_data.filename,
)
popular_negative_samples = test_negative_sampler.get_negative_samples()

#%%
recsys_data.cat2u[0]

for i in tqdm(popular_negative_samples):
    if nsample.get(i) is None:
        print(i)


#%%
print(list(nsample.keys())[:10])
print(list(popular_negative_samples.keys())[:10])
#%%
print(popular_negative_samples[1227024])
for i in tqdm(nsample):
    if popular_negative_samples.get(i) is None:
        nsample[i] = list(set((nsample[i] + popular_negative_samples[1227024])))[:100]
    if len(nsample[i]) < 100:
        nsample[i] = list(set((nsample[i] + popular_negative_samples[i])))[:100]

# print(len(nsample.keys()))
# recsys_data.dataframe.user_id.value_counts()[
#     recsys_data.dataframe.user_id.value_counts() > 3
# ]
# %%



def convert_ids(data_cls, samples):

    converted_samples = {}

    # Iterate over the keys (users) in the input samples dictionary
    for user in tqdm(samples.keys()):
        # For each user, map their ID to the corresponding category using the data_cls.u2cat mapping
        # and store the result as the key in the converted_samples dictionary
        # then map the item's ID to the corresponding category using the data_cls.i2cat mapping
        # and store the result in a list
        converted_samples[data_cls.u2cat[int(user)]] = [
            data_cls.i2cat[int(item)] for item in samples[user]
        ]

    return converted_samples


#%%
max = 0
for i in tqdm(nsample):
    if len(nsample[i]) > 100:
        nsample[i] = nsample[i][:100]

#%%
print(max)
#%%
converted = convert_ids(recsys_data, nsample)


#%%
converted[list(converted.keys())[0]]

nsample[list(nsample.keys())[0]]


# recsys_data.i2cat['59625']

#%%
with open(NEGATIVE_SAMPLE_PATH / "otto_test_nsample_for_carts.pkl", "wb") as f:
    pickle.dump(nsample, f)


#%%
nsample[4985]

# recsys_data.i2cat.keys()
# converted
#%%
# recsys_data.i2cat.keys()
# recsys_data.max_length
# recsys_data.dataframe.user_id.value_counts()
# recsys_data.dataframe[recsys_data.dataframe.user_id == 150567].drop_duplicates

# %%
