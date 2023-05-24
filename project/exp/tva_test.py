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
model_path = "/home/VS6102093/thesis/TVA/logs/beauty.tva4.34_vd128/version_4/checkpoints/epoch=249-step=43000.ckpt"
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

    # SHOW the test result
    trainer = pl.Trainer(gpus=[1], logger=False)
    result = trainer.test(model, dataloaders=test_loader)
    print(result)

    # get the embed
    # total_embed = []

    # for batch in test_loader:
    #     time_seqs = (
    #         batch[f"years"],
    #         batch[f"months"],
    #         batch[f"days"],
    #         batch[f"seasons"],
    #         batch[f"hours"],
    #         batch[f"minutes"],
    #         batch[f"seconds"],
    #         batch[f"dayofweek"],
    #     )
    #     embed = model.embedding(
    #         item_seq=batch[f"item_seq"],
    #         userwise_latent_factor=batch[f"userwise_latent_factor"],
    #         itemwise_latent_factor_seq=batch[f"itemwise_latent_factor_seq"],
    #         time_seqs=time_seqs,
    #     )

    #     total_embed.extend(embed.tolist())

    # # Change total_embed list into numpy array
    # total_embed = np.array(total_embed)
    # print(total_embed.shape)

    # return total_embed


# Exp1. Test the total performance
# test_group_performance(set(recdata.users_seqs.keys()), recdata)


# Exp2. Get different seq_len group and test
seq_groups = generate_seqlen_group(recdata)
for group in tqdm(seq_groups):
    test_group_performance(group, recdata)

# Exp3. Get total embed
# total_embed = test_group_performance(set(recdata.users_seqs.keys()), recdata)
# print(total_embed.shape)
