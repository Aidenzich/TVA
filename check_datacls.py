#%%
import json
import pickle
from src.configs import DATACLASS_PATH, DATA_PATH

data_class_name = "otto_test_cls.pkl"
print(DATACLASS_PATH)
with open(DATACLASS_PATH / data_class_name, "rb") as f:
    recsys_data = pickle.load(f)

with open(DATA_PATH / "otto_test_nsample_for_carts.pkl", "rb") as f:
    nsample = pickle.load(f)

# print(nsample)
# %%
from tqdm import tqdm


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
print(nsample[12899779])


#%%
converted = convert_ids(recsys_data, nsample)


#%%
converted[list(converted.keys())[0]]

nsample[list(nsample.keys())[0]]


# recsys_data.i2cat['59625']


recsys_data.i2cat.keys()
# converted
#%%
recsys_data.i2cat.keys()
# recsys_data.max_length
# recsys_data.dataframe.user_id.value_counts()
# recsys_data.dataframe[recsys_data.dataframe.user_id == 150567].drop_duplicates

# %%
