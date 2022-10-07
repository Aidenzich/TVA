from .BERT4Rec.trainer import BERT4RecTrainer

# import .SASRec
# import .VAECF
# import .TiSASRec

TRAINER_FACTORY = {
    "bert4rec": BERT4RecTrainer,
    # "sasrec": SASRec,
    # "tisasrec": TiSASRec,
    # "vaecf": VAECF,
}
