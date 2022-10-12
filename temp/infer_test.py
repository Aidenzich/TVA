# %%
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import sys

sys.path.append("../")

from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.models.BERT4Rec.model import BERTModel
from src.configs import DATACLASS_PATH, LOG_PATH, OUTPUT_PATH

# Load dataclass and model
with open(DATACLASS_PATH / "data_cls.pkl", "rb") as f:
    myData = pickle.load(f)

mymodel = BERTModel.load_from_checkpoint(
    LOG_PATH / "bert4rec.testing.config/version_0/checkpoints/epoch=29-step=2880.ckpt"
)


#%%

##### INFER ######
torch.cuda.empty_cache()

test_negative_sampler = NegativeSampler(
    train=myData.train_seqs,
    val=myData.val_seqs,
    test=myData.test_seqs,
    user_count=myData.num_users,
    item_count=myData.num_items,
    sample_size=5000,
    seed=12345,
)
test_negative_samples = test_negative_sampler.get_negative_samples()


device = torch.device("cuda:0")
inferset = SequenceDataset(
    mode="inference",
    mask_token=myData.num_items + 1,
    u2seq=myData.train_seqs,
    u2answer=myData.val_seqs,
    max_len=mymodel.max_len,
    negative_samples=test_negative_samples,
)

infer_loader = DataLoader(inferset, batch_size=12, shuffle=False, pin_memory=True)
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
def rollback_ids(data_cls):

    new_dict = {}
    for user in tqdm(predict_result.keys()):
        new_dict[data_cls.cat2u[user]] = [
            data_cls.cat2i[item] for item in predict_result[user]
        ]

    return new_dict


#%%
predict_result_rollback = rollback_ids(myData)
json.dump(
    predict_result_rollback, open(OUTPUT_PATH / "predict_result.json", "w"), indent=2
)

# %%
