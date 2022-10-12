# %%
import pickle
import sys
from src.cli.utils import get_models

sys.path.append("../")
from src.configs import DATACLASS_PATH, LOG_PATH
from src.models import INFER_FACTORY


#%%
from pathlib import Path

import inquirer


def get_dataclass():
    p = Path(DATACLASS_PATH).glob("*")
    data_classes = [x.name for x in p if x.is_file() and "pkl" in str(x)]
    data_classes_path = [x for x in p if x.is_file() and "pkl" in str(x)]

    return data_classes, data_classes_path


data_classes, data_classes_path = get_dataclass()


if __name__ == "__main__":
    data_classes, data_classes_path = get_dataclass()
    models = get_models()
    question = [
        inquirer.List(
            "model",
            message="Which model do you need?",
            choices=models,
        ),
        inquirer.List(
            "data_class",
            message="Which data class do you need? (Check in data/cache/dataclass)",
            choices=data_classes,
        ),
    ]

    answers = inquirer.prompt(question)

    #%%
    with open(DATACLASS_PATH / "data_cls.pkl", "rb") as f:
        myData = pickle.load(f)

    # mymodel = VAECFModel.load_from_checkpoint()

    model_path = (
        LOG_PATH / "vaecf.default.config/version_0/checkpoints/epoch=9-step=9270.ckpt"
    )

    #%%

    # predict_result = infer_vaecf(model_path, myData)
    predict_result = INFER_FACTORY()
    #%%
    list(sorted(myData.cat2u.keys()))

    #%%
    df = myData.dataframe
    #%%
    sorted(df[df.user_id == 1].item_id.unique())
    #%%
    import numpy as np

    np.nonzero(myData.matrix.A[1])

    # infer_vaecf(mymodel, myData)
    # def split(a, n):
    #     k, m = divmod(len(a), n)
    #     return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    # import numpy as np

    # def batch(iterable, n=1):
    #     l = len(iterable)
    #     for ndx in range(0, l, n):
    #         yield iterable[ndx : min(ndx + n, l)]

    # temp = batch(list(myData.u2cat.values()), 12)
    # for i in temp:
    #     print(i)

    ##### INFER ######

    # mymodel.to(device)
    # predict_result: dict = {}
    # ks = 10

    # for batch in tqdm(infer_loader):
    #     seqs, candidates, users = batch
    #     seqs, candidates, users = seqs.to(device), candidates.to(device), users.to(device)
    #     scores = mymodel(seqs)
    #     scores = scores[:, -1, :]  # B x V
    #     scores = scores.gather(1, candidates)  # B x C
    #     rank = (-scores).argsort(dim=1)
    #     predict = candidates.gather(1, rank)
    #     predict_dict = {
    #         u: predict[idx].tolist()[:ks]
    #         for idx, u in enumerate(users.cpu().numpy().flatten())
    #     }
    #     predict_result.update(predict_dict)
    # %%
