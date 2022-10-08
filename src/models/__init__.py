from .BERT4Rec.trainer import bert4rec_train

TRAIN_FACTORY = {
    "bert4rec": bert4rec_train,
}
