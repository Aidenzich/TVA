import torch
from src.datasets.sasrec_dset import SASRecDataset
from src.datasets.negative_sampler import NegativeSampler
from .model import SASRecModel
from src.adapters.lightning_adapter import fit
from src.configs import CACHE_PATH
import numpy as np


def train(model_params, trainer_config, recdata, callbacks=[]):
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

    trainset = SASRecDataset(
        mode="train",
        max_len=model_params["max_len"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        negative_samples=test_negative_samples,
        seed=trainer_config["seed"],
    )

    valset = SASRecDataset(
        mode="eval",
        max_len=model_params["max_len"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.val_seqs,
        negative_samples=test_negative_samples,
        seed=trainer_config["seed"],
    )

    testset = SASRecDataset(
        mode="eval",
        max_len=model_params["max_len"],
        num_items=recdata.num_items,
        mask_token=recdata.num_items + 1,
        u2seq=recdata.train_seqs,
        u2answer=recdata.test_seqs,
        negative_samples=test_negative_samples,
        seed=trainer_config["seed"],
    )

    model = SASRecModel(
        num_items=recdata.num_items,
        num_users=recdata.num_users,
        model_params=model_params,
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

    # NOTE: RuntimeError: CUDA error: device-side assert triggered
    # might be caused by the embedding layer


def infer(ckpt_path, recdata, input_seq, rec_ks):
    # sample_num = int(recdata.num_items)
    sample_num = 100

    device = torch.device("cuda:0")

    torch.cuda.empty_cache()
    model = SASRecModel.load_from_checkpoint(ckpt_path)

    candidates = recdata.dataframe["item_id"].value_counts().index.tolist()[:sample_num]

    candidates[8] = 23757
    # print("candidates num", len(candidates))
    print(candidates[:100])
    candidates = torch.from_numpy(np.array(candidates)).to(device)
    input_seq = torch.from_numpy(np.array(input_seq)).to(device)

    # print("sample items: ", candidates)

    model.to(device)

    scores = model.sasrec.predict(log_seqs=input_seq, item_indices=candidates)
    rank = (-scores).argsort(dim=1)
    topk = rank[:, :rec_ks]
    topk_items = candidates[topk]

    topk = topk.cpu().tolist()

    # print("topk:", topk)
    # print("top items idx:", topk[0][0])
    # print(scores[0, topk[0][0]], scores.max())
    # print("----")

    lowk = rank[:, -rec_ks:]
    lowk_items = candidates[lowk]

    return topk_items, lowk_items
