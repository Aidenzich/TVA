from .BERT4Rec import train_bert4rec, infer_bert4rec
from .VAECF import train_vaecf, infer_vaecf
from .TVA import train_tva, infer_tva
from .TVA2 import train_tva2, infer_tva2
from .BERT4Rec_CNN import train_bert4rec_cnn, infer_bert4rec_cnn

TRAIN_FACTORY = {
    "bert4rec": train_bert4rec,
    "vaecf": train_vaecf,
    "tva": train_tva,
    "tva2": train_tva2,
    "bert4rec_cnn": train_bert4rec_cnn,
}


INFER_FACTORY = {"bert4rec": infer_bert4rec, "vaecf": infer_vaecf, "tva": infer_tva}
