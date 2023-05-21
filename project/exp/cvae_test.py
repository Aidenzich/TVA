# %%
import sys

sys.path.append("../../")

from utils import generate_seqlen_group
from src.models.ContrastVAE.model import ContrastVAEModel
from src.datasets.cvae_dset import CVAEDataset
from src.configs import DATACLASS_PATH
from pytorch_lightning import seed_everything
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

seed_everything(0, workers=True)
# Beauty
# model_path = "/home/VS6102093/thesis/TVA/logs/beauty.contrastvae.re/version_0/checkpoints/epoch=399-step=35200.ckpt"
# dataset = "beauty.pkl"


# toys
# model_path = "/home/VS6102093/thesis/TVA/logs/toys.contrastvae.re/version_0/checkpoints/epoch=399-step=30400.ckpt"
# dataset = "toys.pkl"

# ML1m
model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.contrastvae.nowarm_l2/version_1/checkpoints/epoch=399-step=9600.ckpt"
dataset = "ml1m.pkl"


model = ContrastVAEModel.load_from_checkpoint(model_path)

with open(DATACLASS_PATH / dataset, "rb") as f:
    recdata = pickle.load(f)

recdata.show_info_table()


def test_group_performance(group, recdata) -> None:
    group = set(group)

    user_seqs_4_test = {k: v for k, v in recdata.users_seqs.items() if k in group}

    testset = CVAEDataset(
        mode="test",
        u2seq=user_seqs_4_test,
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        max_len=model.max_len,
    )

    test_loader = DataLoader(
        testset,
        batch_size=2048,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    trainer = pl.Trainer(gpus=[2], logger=False)

    result = trainer.test(model, dataloaders=test_loader)
    print(result)


seq_groups = generate_seqlen_group(recdata)
for group in tqdm(seq_groups):
    test_group_performance(group, recdata)
