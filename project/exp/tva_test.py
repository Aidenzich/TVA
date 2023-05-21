# %%
import sys

sys.path.append("../../")

from utils import generate_seqlen_group
from src.models.TVA4.model import TVAModel
from src.datasets.tva_dset import TVASequenceDataset
from src.configs import DATACLASS_PATH
import numpy as np
import pickle
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

seed_everything(0, workers=True)
user_latent_factor = None

# # Beauty
model_path = "/home/VS6102093/thesis/TVA/logs/beauty.tva4.34_vd128/version_1/checkpoints/epoch=249-step=43000.ckpt"
latent_path = "/home/VS6102093/thesis/TVA/logs/beauty.vaeicf.d128/version_0/latent_factor/encode_result.npy"
dataset = "beauty.pkl"

# Toys
# model_path = "/home/VS6102093/thesis/TVA/logs/toys.tva4.n24_vd256/version_1/checkpoints/epoch=249-step=31000.ckpt"
# latent_path = "/home/VS6102093/thesis/TVA/logs/toys.vaeicf.d256/version_0/latent_factor/encode_result.npy"
# dataset = "toys.pkl"


# ML1m
# model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.tva4.28_len80_n3_m0.4_best/version_0/checkpoints/epoch=249-step=67000.ckpt"
# latent_path = "/home/VS6102093/thesis/TVA/logs/ml1m.vaeicf.d256/version_0/latent_factor/encode_result.npy"
# dataset = "ml1m.pkl"

model = TVAModel.load_from_checkpoint(model_path)
model.eval()
item_latent_factor = np.load(latent_path)

with open(DATACLASS_PATH / dataset, "rb") as f:
    recdata = pickle.load(f)

recdata.show_info_table()


def test_group_performance(group, recdata) -> None:
    group = set(group)

    # filter out data not in seq_len_10
    train_seqs = {k: v for k, v in recdata.train_seqs.items() if k in group}
    train_timeseqs = {k: v for k, v in recdata.train_timeseqs.items() if k in group}
    val_seqs = {k: v for k, v in recdata.val_seqs.items() if k in group}
    val_timeseqs = {k: v for k, v in recdata.val_timeseqs.items() if k in group}
    test_seqs = {k: v for k, v in recdata.test_seqs.items() if k in group}
    test_timeseqs = {k: v for k, v in recdata.test_timeseqs.items() if k in group}

    testset = TVASequenceDataset(
        mode="eval",
        max_len=model.max_len,
        mask_token=recdata.num_items + 1,
        u2seq=train_seqs,
        u2answer=test_seqs,
        u2answer_time=test_timeseqs,
        u2timeseq=train_timeseqs,
        num_items=recdata.num_items,
        # Add val item into item seqence
        u2val=val_seqs,
        u2val_time=val_timeseqs,
        user_latent_factor=user_latent_factor,
        seed=0,
        item_latent_factor=item_latent_factor,
    )

    test_loader = DataLoader(
        testset,
        batch_size=5096,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    trainer = pl.Trainer(gpus=[1], logger=False)

    result = trainer.test(model, dataloaders=test_loader)
    print(result)


# test_group_performance(set(recdata.users_seqs.keys()), recdata)


seq_groups = generate_seqlen_group(recdata)
for group in tqdm(seq_groups):
    test_group_performance(group, recdata)

# %%
