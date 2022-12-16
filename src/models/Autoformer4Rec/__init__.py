from torch import ne
from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from .model import BERTModel
from src.adapters.lightning_adapter import fit


def train(model_params, trainer_config, recdata, callbacks=[]):
    # FIXME This can be store in the RecData class
    test_negative_sampler = NegativeSampler(
        train=recdata.train_seqs,
        val=recdata.val_seqs,
        test=recdata.test_seqs,
        item_count=recdata.num_items,
        sample_size=trainer_config["sample_size"],
        method=trainer_config["sample_method"],
        seed=trainer_config["seed"],
        dataclass_name=recdata.filename,
    )

    test_negative_samples = test_negative_sampler.get_negative_samples()

    trainset = SequenceDataset(
        mode="train",
        max_len=model_params["max_len"],
        mask_prob=model_params["mask_prob"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        seed=trainer_config["seed"],
    )

    valset = SequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        negative_samples=test_negative_samples,
    )

    testset = SequenceDataset(
        mode="eval",
        max_len=model_params["max_len"],
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        negative_samples=test_negative_samples,
    )

    model = BERTModel(
        hidden_size=model_params["hidden_size"],
        num_items=recdata.num_items,
        n_layers=model_params["n_layers"],
        dropout=model_params["dropout"],
        heads=model_params["heads"],
        max_len=model_params["max_len"],
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

    inferset = SequenceDataset(
        mode="inference",
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,  # TODO 把 inference 的 train_seqs 改成新資料(注意要把id都轉成新的)
        max_len=model.max_len,
        negative_samples=negative_samples,
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
