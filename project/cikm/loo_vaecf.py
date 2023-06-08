# %%

import numpy as np
from tqdm import tqdm
import sys
import pickle
from utils import recall_at_k, ndcg_at_k

sys.path.append("../../")
from src.configs import DATACLASS_PATH, LOG_PATH
from src.models.VAECF import VAECFModel
from src.datasets.vaecf_dset import VAECFDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm


dataset = "beauty.pkl"
dataset = "toys.pkl"
dataset = "ml1m.pkl"
device = torch.device("cuda:0")

if dataset == "beauty.pkl":
    model_path = "/home/VS6102093/thesis/TVA/logs/beauty.vaecf.d128/version_0/checkpoints/epoch=99-step=4300.ckpt"
if dataset == "toys.pkl":
    # model_path = "/home/VS6102093/thesis/TVA/logs/toys.vaecf.d128/version_0/checkpoints/epoch=25-step=988.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/toys.vaecf.d128/version_1/checkpoints/epoch=99-step=3800.ckpt"
    model_path = "/home/VS6102093/thesis/TVA/logs/toys.vaecf.d256/version_1/checkpoints/epoch=99-step=3800.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/toys.vaecf.d512/version_0/checkpoints/epoch=24-step=950.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/toys.vaecf.d512/version_1/checkpoints/epoch=99-step=3800.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/toys.vaecf.d512_beta0.2/version_0/checkpoints/epoch=99-step=3800.ckpt"

if dataset == "ml1m.pkl":
    # model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaecf.d256/version_0/checkpoints/epoch=99-step=1200.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaecf.d512/version_0/checkpoints/epoch=99-step=1200.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaecf.d128/version_0/checkpoints/epoch=99-step=1200.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaecf.d256_as256/version_1/checkpoints/epoch=99-step=1200.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaecf.d256_as1024/version_0/checkpoints/epoch=99-step=1200.ckpt"
    # model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaecf.d256_as256_beta1.0/version_0/checkpoints/epoch=99-step=1200.ckpt"
    model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaecf.d256_as256_beta0.4/version_30/checkpoints/epoch=99-step=1200.ckpt"


with open(DATACLASS_PATH / dataset, "rb") as f:
    recdata = pickle.load(f)

recdata.show_info_table()

inferset = VAECFDataset(recdata.matrix)
model = VAECFModel.load_from_checkpoint(model_path)
infer_loader = DataLoader(
    inferset, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True
)


model.to(device)
predict_result: dict = {}

user_id = 0

model.eval()
with torch.no_grad():
    for batch in tqdm(infer_loader):
        batch = batch.to(device)
        z_u, z_sigma = model.vae.encode(batch)
        y = model.vae.decode(z_u)

        z_u, z_sigma = z_u.cpu().numpy().astype(
            np.float16
        ), z_sigma.cpu().numpy().astype(np.float16)

        seen = batch != 0
        y[seen] = 0

        top_k = y.topk(30, dim=1)[1]

        y = y.cpu().numpy().astype(np.float16)

        for idx in range(batch.shape[0]):
            predict_result[user_id] = top_k[idx].tolist()
            user_id = user_id + 1


sum_recall_5, sum_recall_10, sum_recall_20 = 0, 0, 0
sum_ndcg_5, sum_ndcg_10, sum_ndcg_20 = 0, 0, 0
for user in tqdm(recdata.test_seqs):
    pred_list = predict_result[user]
    true_list = recdata.test_seqs[user]
    recall_5 = recall_at_k(true_list=true_list, pred_list=pred_list, k=5)
    recall_10 = recall_at_k(true_list=true_list, pred_list=pred_list, k=10)
    recall_20 = recall_at_k(true_list=true_list, pred_list=pred_list, k=20)

    ndcg_5 = ndcg_at_k(true_list=true_list, pred_list=pred_list, k=5)
    ndcg_10 = ndcg_at_k(true_list=true_list, pred_list=pred_list, k=10)
    ndcg_20 = ndcg_at_k(true_list=true_list, pred_list=pred_list, k=20)

    sum_recall_5 += recall_5
    sum_recall_10 += recall_10
    sum_recall_20 += recall_20
    sum_ndcg_5 += ndcg_5
    sum_ndcg_10 += ndcg_10
    sum_ndcg_20 += ndcg_20

ndcg_5 = sum_ndcg_5 / len(recdata.test_seqs)
ndcg_10 = sum_ndcg_10 / len(recdata.test_seqs)
ndcg_20 = sum_ndcg_20 / len(recdata.test_seqs)
recall_5 = sum_recall_5 / len(recdata.test_seqs)
recall_10 = sum_recall_10 / len(recdata.test_seqs)
recall_20 = sum_recall_20 / len(recdata.test_seqs)


print(recall_5)
print(recall_10)
print(recall_20)
print(ndcg_5)
print(ndcg_10)
print(ndcg_20)
