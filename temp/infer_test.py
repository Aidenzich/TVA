# %%
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys

sys.path.append("../")

from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.models.BERT4Rec.model import BERTModel
from src.configs import DATA_PATH, LOG_PATH

#%%
max_len = 128
with open(DATA_PATH / "data_cls.pkl", "rb") as f:
    myData = pickle.load(f)

#%%
##### INFER ######
torch.cuda.empty_cache()

test_negative_sampler = NegativeSampler(
    train=myData.train_seqs,
    val=myData.val_seqs,
    test=myData.test_seqs,
    user_count=myData.num_users,
    item_count=myData.num_items,
    sample_size=120,
    seed=12345,
)
test_negative_samples = test_negative_sampler.get_negative_samples()


device = torch.device("cuda:0")
inferset = SequenceDataset(
    mode="inference",
    mask_token=myData.num_items + 1,
    u2seq=myData.train_seqs,
    u2answer=myData.val_seqs,
    max_len=max_len,
    negative_samples=test_negative_samples,
)

infer_loader = DataLoader(inferset, batch_size=12, shuffle=False, pin_memory=True)

mymodel = BERTModel.load_from_checkpoint(
    LOG_PATH / "lightning_logs/version_0/checkpoints/epoch=9-step=1000.ckpt"
)


mymodel.to(device)
predict_result: dict = {}
ks = 10

for batch in tqdm(infer_loader):
    seqs, candidates, users = batch
    seqs, candidates, users = seqs.to(device), candidates.to(device), users.to(device)
    scores = mymodel(seqs)
    scores = scores[:, -1, :]  # B x V
    scores = scores.gather(1, candidates)  # B x C
    rank = (-scores).argsort(dim=1)
    predict = candidates.gather(1, rank)
    predict_dict = {
        u: predict[idx].tolist()[:ks]
        for idx, u in enumerate(users.cpu().numpy().flatten())
    }
    predict_result.update(predict_dict)
# %%
