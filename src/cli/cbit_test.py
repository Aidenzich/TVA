#%%
import sys
sys.path.append("../../")
# from src.models.TVA4.model import TVAModel
from src.models.CBiT.model import CBiTModel

from src.datasets.cbit_dset import CBiTDataset
from src.configs import DATACLASS_PATH
import numpy as np
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
model = CBiTModel.load_from_checkpoint("/home/VS6102093/thesis/TVA/logs/beauty.cbit.re/version_2/checkpoints/epoch=249-step=43000.ckpt")
user_latent_factor = None
item_latent_factor = np.load("/home/VS6102093/thesis/TVA/logs/beauty.vaeicf.d128/version_0/latent_factor/encode_result.npy")

with open(DATACLASS_PATH / "beauty.pkl", "rb") as f:
    recdata = pickle.load(f)

recdata.show_info_table()


#%%

freq_20 = []

freq_40 = []
freq_large = []

freq = {}
for i in tqdm(recdata.test_seqs):
    item = recdata.test_seqs[i][0]
    if freq.get(item, None) is None:
        freq[item] = 1
    else:
        freq[item] += 1
        
for i in tqdm(recdata.test_seqs):
    item = recdata.test_seqs[i][0]


    if freq[item] <= 20:
        freq_20.append(i)
    elif freq[item] <= 40:
        freq_40.append(i)    
    else:
        freq_large.append(i)

print("=="*50)
# print(len(freq_10))
print(len(freq_20))
# print(len(freq_30))
print(len(freq_40))
print(len(freq_large))
print("=="*50)

#%%

seq_len_10 = []
seq_len_20 = []
seq_len_30 = []
seq_len_40 = []
seq_len_large = []


for i in tqdm(recdata.users_seqs):
    seq_len = len(recdata.users_seqs[i]) 
    seq_len = seq_len - 1
    if seq_len <= 10:
        seq_len_10.append(i)
    elif seq_len <= 20:
        seq_len_20.append(i)
    elif seq_len <= 30:
        seq_len_30.append(i)
    elif seq_len <= 40:
        seq_len_40.append(i)
    else:
        seq_len_large.append(i)

#%%
print("=="*50)
print(len(seq_len_10))
print(len(seq_len_20))
print(len(seq_len_30))
print(len(seq_len_40))
print(len(seq_len_large))
print("=="*50)



#%%

train_seqs = recdata.train_seqs
train_timeseqs = recdata.train_timeseqs
val_seqs = recdata.val_seqs
val_timeseqs = recdata.val_timeseqs
test_seqs = recdata.test_seqs
test_timeseqs = recdata.test_timeseqs

# Convert seq_len_10 to a set for faster lookups
seq_len = set(seq_len_10)
# seq_len = set(freq_20)

# filter out data not in seq_len_10
train_seqs = {k: v for k, v in train_seqs.items() if k in seq_len}
train_timeseqs = {k: v for k, v in train_timeseqs.items() if k in seq_len}
val_seqs = {k: v for k, v in val_seqs.items() if k in seq_len}
val_timeseqs = {k: v for k, v in val_timeseqs.items() if k in seq_len}
test_seqs = {k: v for k, v in test_seqs.items() if k in seq_len}
test_timeseqs = {k: v for k, v in test_timeseqs.items() if k in seq_len}


#%%
len(train_seqs)

#%%


testset = CBiTDataset(
    mode="eval",
    max_len=model.max_len,
    mask_token=recdata.num_items + 1,
    num_items=recdata.num_items,
    u2seq=train_seqs,
    u2val=val_seqs,
    u2answer=test_seqs,    
)



test_loader = DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    pin_memory=False,
    num_workers=1,
)

trainer = pl.Trainer(gpus=1)

result = trainer.test(model, dataloaders=test_loader)
# %%
