#%%
import pandas as pd
import cornac
from torch import seed
import numpy as np
import sys

sys.path.append("../")
from src.datasets.common import RecsysData
from torch.utils.data import Dataset, DataLoader
import torch

data = pd.read_pickle("../data/carrefour.pkl")
vc = data.user_id.value_counts()
user10index = vc[vc > 10].index
data = data[data.user_id.isin(user10index)]
data.rename(columns={"quantity": "rating"}, inplace=True)
# print(user10index)
# %%
# cdata = pd.read_pickle("../data/trade.pkl")
# trainset = cornac.data.Dataset.from_uir(cdata.values, seed=123)
# trainset.matrix.shape[1]
# temp = trainset.user_iter(123, shuffle=False)
# for batch_id, u_ids in enumerate(trainset.user_iter(12, shuffle=False)):
#     print(u_ids)
#     u_batch = trainset.matrix[u_ids, :]
#     u_batch.data = np.ones(len(u_batch.data))  # Binarize data
#     u_batch = u_batch.A
#     u_batch = torch.tensor(u_batch, dtype=torch.float32)
#     print(u_batch.shape)
#     break


# %%
myData = RecsysData(data)
myData.prepare_matrix()
#%%

temp = myData.matrix[0, :]


#%%
myData.matrix.shape
#%%
temp.A.tolist()

#%%


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        buy_items = self.data[idx, :].A[0]
        return torch.tensor(buy_items, dtype=torch.float32)


dset = MyDataset(myData.matrix)
train_loader = DataLoader(dset, batch_size=128, shuffle=False, num_workers=0)
for u_batch in train_loader:
    print(len(u_batch))
    print(u_batch.shape)
    # print(i.shape)
    break


# %%
myData.matrix[[0, 1], :].shape
# %%
from src.models.VAECF.model import VAE

data_dim = myData.matrix.shape[1]

vae = VAE(
    z_dim=10,
    ae_structure=[data_dim] + [20],
    activation_function="tanh",
    likelihood="mult",
)

# %%
u_batch_, mu, logvar = vae(u_batch)

# %%
print(u_batch_.shape, mu.shape, logvar.shape)

loss = vae.loss(u_batch, u_batch_, mu, logvar, 0.1)
# %%
loss

# %%
