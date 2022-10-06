from .BERT4Rec.train import BERT4RecTrain

# import .SASRec
# import .VAECF
# import .TiSASRec

MODEL_FACTORY = {
    "bert4rec": BERT4RecTrain,
    # "sasrec": SASRec,
    # "tisasrec": TiSASRec,
    # "vaecf": VAECF,
}
