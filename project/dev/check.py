# %%
import sys

sys.path.append("../../")

import matplotlib.pyplot as plt


import pickle
from src.configs import DATACLASS_PATH, LOG_PATH
from tqdm import tqdm

# %%
data_class_name = "ratings_toys_and_games_5_test.pkl"


with open(DATACLASS_PATH / data_class_name, "rb") as f:
    recsys_data = pickle.load(f)

# %%
data_class_name2 = "toys.pkl"


with open(DATACLASS_PATH / data_class_name2, "rb") as f:
    recsys_data2 = pickle.load(f)

# %%
recsys_data.show_info_table()
recsys_data2.show_info_table()
count = 0
for i in recsys_data2.users_seqs:
    if recsys_data2.users_seqs[i] == recsys_data.users_seqs[i] != True:
        count += 1

print(count)

# %%
import numpy as np

with open(
    LOG_PATH
    / "ratings_beauty_5.vaecf.default/version_0/latent_factor/decode_result.npy",
    "rb",
) as f:
    vae_result = np.load(f)


# %%
len(recsys_data.users_seqs)

user_item_matrix = np.zeros_like(vae_result)
print(user_item_matrix.shape)

for u in range(recsys_data.num_users):
    # draw user's bought items count
    for i in recsys_data.val_seqs[u]:
        user_item_matrix[u, i] = 1

    for i in recsys_data.test_seqs[u]:
        user_item_matrix[u, i] = 1


# %%
count = 0

for u in range(recsys_data.num_users):
    x = user_item_matrix[u]

    vae_x = vae_result[u]
    # softmax_x = np.exp(x) / np.sum(np.exp(x))
    softmax_vae_x = np.exp(vae_x) / np.sum(np.exp(vae_x))

    top_indices = np.argsort(softmax_vae_x)[-100:]
    top_values = softmax_vae_x[top_indices]
    softmax_vae_x[top_indices] = 0.5

    plt.plot(x)
    plt.plot(softmax_vae_x)
    plt.title(f"Image {u}")
    # plt.show()

    # save image
    plt.savefig(f"temp/image_{u}.png")
    plt.close()
    count += 1
    # if count > 10:
    #     break


# %%
