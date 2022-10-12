from .BERT4Rec import train_bert4rec, infer_bert4rec
from .VAECF import train_vaecf, infer_vaecf

TRAIN_FACTORY = {
    "bert4rec": train_bert4rec,
    "vaecf": train_vaecf,
}


INFER_FACTORY = {"bert4rec": infer_bert4rec, "vaecf": infer_vaecf}
