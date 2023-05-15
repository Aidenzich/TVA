import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm


def generate_seqlen_group(recdata):
    seq_len_10 = []
    seq_len_20 = []
    seq_len_30 = []
    seq_len_40 = []
    seq_len_large = []

    for i in tqdm(recdata.users_seqs):
        seq_len = len(recdata.users_seqs[i])
        seq_len = seq_len - 1
        if seq_len <= 10:
            seq_len_10.append(i)
        elif seq_len <= 20:
            seq_len_20.append(i)
        elif seq_len <= 30:
            seq_len_30.append(i)
        elif seq_len <= 40:
            seq_len_40.append(i)
        else:
            seq_len_large.append(i)

    print("==" * 50)
    print(len(seq_len_10))
    print(len(seq_len_20))
    print(len(seq_len_30))
    print(len(seq_len_40))
    print(len(seq_len_large))
    print("==" * 50)

    return seq_len_10, seq_len_20, seq_len_30, seq_len_40, seq_len_large
