from .BERT4Rec import train_bert4rec, infer_bert4rec
from .VAECF import train_vaecf, infer_vaecf
from .TVA2 import train_tva2, infer_tva2


TRAIN_FACTORY = {
    "bert4rec": train_bert4rec,
    "vaecf": train_vaecf,
    "tva2": train_tva2,
}


INFER_FACTORY = {"bert4rec": infer_bert4rec, "vaecf": infer_vaecf, "tva": infer_tva2}
