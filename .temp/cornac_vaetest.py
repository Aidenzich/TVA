#%%
import cornac
from cornac.datasets import citeulike
from cornac.eval_methods import RatioSplit
import sys

sys.path.append("../")
from src.configs.paths import DATA_PATH
import pickle

# Load user-item feedback
data = citeulike.load_feedback()


with open(DATA_PATH / "data_cls.pkl", "rb") as f:
    recsys_data = pickle.load(f)

#%%
# data = recsys_data.dataframe[["user_id", "item_id", "rating"]].values.tolist()


#%%

# Instantiate an evaluation method to split data into train and test sets.
ratio_split = RatioSplit(
    data=data,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=0,
)

# Instantiate the VAECF model
vaecf = cornac.models.VAECF(
    k=10,
    autoencoder_structure=[20],
    act_fn="tanh",
    likelihood="mult",
    n_epochs=100,
    batch_size=12,
    learning_rate=0.001,
    beta=1.0,
    seed=123,
    use_gpu=True,
    verbose=True,
)

# Instantiate evaluation measures
rec_20 = cornac.metrics.Recall(k=20)

# Put everything together into an experiment and run it
# cornac.Experiment(
#     eval_method=ratio_split,
#     models=[vaecf],
#     metrics=[rec_20],
#     user_based=True,
# ).run()


ratio_split
#%%
from numpy import random

train_size = int(len(data) * 0.8)
test_size = int(len(data) * 0.1)
test_size = len(data) - (train_size + test_size)


data_idx = random.permutation(len(data))
train_idx = data_idx[:train_size]
test_idx = data_idx[-test_size:]
val_idx = data_idx[train_size:-test_size]

# %%
val_idx

import time


def evaluate(self, model, metrics, user_based, show_validation=True):
    if self.train_set is None:
        raise ValueError("train_set is required but None!")
    if self.test_set is None:
        raise ValueError("test_set is required but None!")

    self._reset()
    self._organize_metrics(metrics)

    ###########
    # FITTING #
    ###########
    if self.verbose:
        print("\n[{}] Training started!".format(model.name))

    start = time.time()
    model.fit(self.train_set, self.val_set)
    train_time = time.time() - start

    ##############
    # EVALUATION #
    ##############
    if self.verbose:
        print("\n[{}] Evaluation started!".format(model.name))

    start = time.time()
    test_result = self._eval(
        model=model,
        test_set=self.test_set,
        val_set=self.val_set,
        user_based=user_based,
    )
    test_time = time.time() - start
    test_result.metric_avg_results["Train (s)"] = train_time
    test_result.metric_avg_results["Test (s)"] = test_time

    val_result = None
    if show_validation and self.val_set is not None:
        start = time.time()
        val_result = self._eval(
            model=model, test_set=self.val_set, val_set=None, user_based=user_based
        )
        val_time = time.time() - start
        val_result.metric_avg_results["Time (s)"] = val_time

    return test_result, val_result
