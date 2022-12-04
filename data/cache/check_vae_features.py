#%%
import numpy as np

variance = np.load("./beauty_latent_factor.npy")

# %%
print(variance.shape)


import matplotlib.pyplot as plt

plt.plot([i for i in range(512)], variance[0])
print(variance[0].mean())
print(variance[0].var())

#%%
plt.plot([i for i in range(512)], variance[1])
print(variance[0].mean())
print(variance[0].var())
# %%
plt.plot([i for i in range(512)], variance[3])
print(variance[0].mean())
print(variance[0].var())
# %%
plt.plot([i for i in range(512)], variance[4])
print(variance[0].mean())
print(variance[0].var())
# %%
plt.plot([i for i in range(512)], variance[5])
print(variance[0].mean())
print(variance[0].var())
