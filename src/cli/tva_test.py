from ..models.TVA4.model import TVAModel
from ..datasets.tva_dset import TVASequenceDataset
from ..configs import DATACLASS_PATH
import numpy as np
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

model = TVAModel.load_from_checkpoint("/home/VS6102093/thesis/TVA/logs/beauty.tva4.34_vd128/version_1/checkpoints/epoch=249-step=43000.ckpt")
user_latent_factor = None
item_latent_factor = np.load("/home/VS6102093/thesis/TVA/logs/beauty.vaeicf.d128/version_0/latent_factor/encode_result.npy")

with open(DATACLASS_PATH / "beauty.pkl", "rb") as f:
    recdata = pickle.load(f)

recdata.show_info_table()

print(model.max_len)
testset = TVASequenceDataset(
    mode="eval",
    max_len=model.max_len,
    mask_token=recdata.num_items + 1,
    u2seq=recdata.train_seqs,
    u2answer=recdata.test_seqs,
    u2answer_time=recdata.test_timeseqs,
    u2timeseq=recdata.train_timeseqs,
    num_items=recdata.num_items,
    # Add val item into item seqence
    u2val=recdata.val_seqs,
    u2val_time=recdata.val_timeseqs,
    user_latent_factor=user_latent_factor,
    seed=0,
    item_latent_factor=item_latent_factor,
)

test_loader = DataLoader(
    testset,
    batch_size=128,
    shuffle=False,
    pin_memory=False,
    num_workers=1,
)

trainer = pl.Trainer(gpus=1)

trainer.test(model, dataloaders=test_loader)