# %%
import sys

sys.path.append("../../")

from utils import generate_seqlen_group
from src.models.CBiT.model import CBiTModel
from src.datasets.cbit_dset import CBiTDataset
from src.configs import DATACLASS_PATH
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Toys
model_path = "/home/VS6102093/thesis/TVA/logs/toys.cbit.re/version_2/checkpoints/epoch=249-step=31000.ckpt"
dataset = "toys.pkl"


# ML-1M
model_path = "/home/VS6102093/thesis/TVA/logs/ml1m.cbit.re/version_3/checkpoints/epoch=249-step=82750.ckpt"
dataset = "ml1m.pkl"


# Beauty


model = CBiTModel.load_from_checkpoint(model_path)

with open(DATACLASS_PATH / dataset, "rb") as f:
    recdata = pickle.load(f)

recdata.show_info_table()


def test_group_performance(group, recdata) -> None:
    group = set(group)
    train_seqs = {k: v for k, v in recdata.train_seqs.items() if k in group}
    train_timeseqs = {k: v for k, v in recdata.train_timeseqs.items() if k in group}
    val_seqs = {k: v for k, v in recdata.val_seqs.items() if k in group}
    val_timeseqs = {k: v for k, v in recdata.val_timeseqs.items() if k in group}
    test_seqs = {k: v for k, v in recdata.test_seqs.items() if k in group}
    test_timeseqs = {k: v for k, v in recdata.test_timeseqs.items() if k in group}

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
        batch_size=2048,
        shuffle=False,
        pin_memory=False,
        num_workers=1,
    )

    trainer = pl.Trainer(gpus=[0], logger=False)

    result = trainer.test(model, dataloaders=test_loader)
    print(result)


seq_groups = generate_seqlen_group(recdata)

for group in seq_groups:
    test_group_performance(group, recdata)

# %%
