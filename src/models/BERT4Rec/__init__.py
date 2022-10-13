from src.datasets.seq_dset import SequenceDataset
from src.datasets.negative_sampler import NegativeSampler
from src.models.BERT4Rec.model import BERTModel
from src.adapters.lightning_adapter import fit


def train_bert4rec(model_params, trainer_config, recdata, callbacks=[]):
    # FIXME This can be store in the RecData class
    test_negative_sampler = NegativeSampler(
        train=recdata.train_seqs,
        val=recdata.val_seqs,
        test=recdata.test_seqs,
        user_count=recdata.num_users,
        item_count=recdata.num_items,
        sample_size=trainer_config["sample_size"],
        method="random",
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
        trainer_config=trainer_config,
        model_params=model_params,
        testset=testset,
        callbacks=callbacks,
    )


def infer_bert4rec(ckpt_path, recdata, rec_ks=10, negative_samples=None):
    """rec k is the number of items to recommend"""
    ##### INFER ######
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = torch.device("cuda:0")

    torch.cuda.empty_cache()

    model = BERTModel.load_from_checkpoint(ckpt_path)

    if negative_samples == None:
        negative_sampler = NegativeSampler(
            train=recdata.train_seqs,
            val=recdata.val_seqs,
            test=recdata.test_seqs,
            user_count=recdata.num_users,
            item_count=recdata.num_items,
            sample_size=3000
            if recdata.num_items * 0.8 > 3000
            else int(recdata.num_items * 0.8),
            dataclass_name=recdata.filename,
            seed=12345,
        )
        negative_samples = negative_sampler.get_negative_samples()

    inferset = SequenceDataset(
        mode="inference",
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        max_len=model.max_len,
        negative_samples=negative_samples,
    )

    infer_loader = DataLoader(inferset, batch_size=12, shuffle=False, pin_memory=True)
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
