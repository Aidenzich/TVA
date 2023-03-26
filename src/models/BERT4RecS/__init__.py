from src.datasets.bert_dset import BertDataset
from src.models.BERT4Rec.model import BERTModel
from src.adapters.lightning_adapter import fit
from typing import List
from tqdm import tqdm


def train(model_params, trainer_config, recdata, callbacks=[]) -> None:
    slided_u2train_seqs = {}

    for u in tqdm(recdata.train_seqs):
        slided_user_seqs = get_slidewindow(
            recdata.train_seqs[u], model_params["max_len"], step=1
        )

        for idx, seqs in enumerate(slided_user_seqs):
            slided_u2train_seqs[str(u) + "." + str(idx)] = seqs

    print(f"Before sliding window data num: {len(recdata.train_seqs)}")
    print(f"After sliding window data num: {len(slided_u2train_seqs)}")

    trainset = BertDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=slided_u2train_seqs,
        seed=trainer_config["seed"],
    )

    valset = BertDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
    )

    testset = BertDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        u2seq=recdata.train_seqs,
        u2val=recdata.val_seqs,
        u2answer=recdata.test_seqs,
    )

    model = BERTModel(
        num_items=recdata.num_items,
        model_params=model_params,
        trainer_config=trainer_config,
    )

    fit(
        model=model,
        trainset=trainset,
        valset=valset,
        testset=testset,
        trainer_config=trainer_config,
        model_params=model_params,
        callbacks=callbacks,
    )


def infer(ckpt_path, recdata, rec_ks=10, negative_samples=None):
    """
    rec k is the number of items to recommend
    """

    ##### INFER ######
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = torch.device("cuda:0")

    torch.cuda.empty_cache()

    model = BERTModel.load_from_checkpoint(ckpt_path)

    if negative_samples == None:
        sample_num = int(recdata.num_items * 0.2)

        if sample_num > 10000:
            print(
                "Sample num is too large, set to 10000. (Due to 2070's memory limitation)"
            )
            sample_num = 10000

        negative_samples = {}

        sample_items = (
            recdata.dataframe["item_id"].value_counts().index.tolist()[:sample_num]
        )
        for u in range(recdata.num_users):
            negative_samples[u] = sample_items

    # Check if the the trainset user are in the negative samples

    train_keys = list(recdata.train_seqs.keys())
    for k in tqdm(train_keys):
        if negative_samples.get(k, None) == None:
            print(k, "is not in the negative samples")
            recdata.train_seqs.pop(k, None)

    inferset = BertDataset(
        mode="inference",
        mask_token=recdata.num_items + 1,
        num_items=recdata.num_items,
        u2seq=recdata.train_seqs,  # TODO 把 inference 的 train_seqs 改成新資料(注意要把id都轉成新的)
        max_len=model.max_len,
    )

    infer_loader = DataLoader(inferset, batch_size=4, shuffle=False, pin_memory=False)

    model.to(device)
    predict_result: dict = {}
    with torch.no_grad():
        for batch in tqdm(infer_loader):

            seqs, candidates, users = batch
            seqs, candidates, users = (
                seqs.to(device),
                candidates.to(device),
                users.to(device),
            )

            scores = model(seqs)
            scores = scores[:, -1, :]  # B x V
            scores = scores.gather(1, candidates)  # B x C
            rank = (-scores).argsort(dim=1)
            predict = candidates.gather(1, rank)

            predict_dict = {
                u: predict[idx].tolist()[:rec_ks]
                for idx, u in enumerate(users.cpu().numpy().flatten())
            }
            predict_result.update(predict_dict)

    return predict_result


def get_slidewindow(user_seq, max_len, step=10) -> List[int]:
    user_sliding_seqs = []

    if isinstance(user_seq[1], tuple):
        seq = [x[0] for x in user_seq]
    else:
        seq = user_seq

    seq_len = len(seq)
    beg_idx = list(range(seq_len - max_len, 0, -step))
    beg_idx.append(0)
    for i in beg_idx:
        temp = seq[i : i + max_len]
        user_sliding_seqs.append(temp)

    return user_sliding_seqs
