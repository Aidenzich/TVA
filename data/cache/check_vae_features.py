#%%
import numpy as np

variance = np.load("./variance.npy")

# %%
variance.shape


#%%
import torch

t_tensor = torch.LongTensor([[j for j in range(3)] for _ in range(3)])

# %%
t_tensor
# %%
torch.cat([t_tensor, t_tensor], dim=-1)