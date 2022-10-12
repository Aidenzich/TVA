from .BERT4Rec.trainer import bert4rec_train
from .VAECF.trainer import vaecf_train

TRAIN_FACTORY = {
    "bert4rec": bert4rec_train,
    "vaecf": vaecf_train,
}


INFER_FACTORY = {}
