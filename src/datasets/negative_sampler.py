from abc import *
from pathlib import Path
import pickle
import numpy as np
from tqdm import trange, tqdm
from collections import Counter
from src.configs import NEGATIVE_SAMPLE_PATH, RED_COLOR, END_COLOR


class NegativeSampler(metaclass=ABCMeta):
    """_summary_
    Negativesampler is used to generate negative samples for training.
    Negative samples is a dictionary, key is user id, value is a list of negative samples.
    """

    def __init__(
        self,
        train,
        val,
        test,
        item_count,
        sample_size,
        seed,
        dataclass_name,
        method="random",
    ) -> None:
        self.dataclass_name = dataclass_name.lower()
        self.train = train
        self.val = val
        self.test = test
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.method = method
        if self.sample_size > self.item_count:
            raise ValueError(
                RED_COLOR
                + f"Sample size {self.sample_size} is larger than item nums {self.item_count}, please check your config"
                + END_COLOR
            )

    def items_by_popularity(self):
        popularity = Counter()
        for user in self.train.keys():
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print("Negatives samples exist. Loading...")
            negative_samples = pickle.load(savefile_path.open("rb"))
            print("Negatives samples Loaded.")

            return negative_samples
        print("Negative samples don't exist. Generating.")
        negative_samples = self.generate_negative_samples()
        print("Saving negative samples")
        with savefile_path.open("wb") as f:
            pickle.dump(negative_samples, f)

        return negative_samples

    def generate_negative_samples(self):
        if self.method == "random":
            return self._generate_random_negative_samples()
        if self.method == "popular":
            return self._generate_popular_negative_samples()
        else:
            raise ValueError("Invalid method")

    def _generate_random_negative_samples(self):
        assert self.seed is not None, (
            RED_COLOR + "Specify seed for random sampling" + END_COLOR
        )
        np.random.seed(self.seed)
        negative_samples = {}
        print("Sampling random negative items")
        for user in tqdm(self.train.keys()):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count) + 1
                wait_patience = 0
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                    if wait_patience > 100:
                        assert False, (
                            RED_COLOR
                            + "Too many patience. Please check your config, sample_size might be too large"
                            + END_COLOR
                        )
                    wait_patience += 1

                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def _generate_popular_negative_samples(self):
        popular_items = self.items_by_popularity()

        negative_samples = {}
        print("Sampling popular negative items")
        for user in tqdm(self.train.keys()):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            samples = []
            for item in popular_items:
                if len(samples) == self.sample_size:
                    break
                if item in seen:
                    continue
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def _get_save_path(self) -> Path:
        filename = "{}.{}-sample_size{}-seed{}.pkl".format(
            self.dataclass_name, self.method, self.sample_size, self.seed
        )

        return NEGATIVE_SAMPLE_PATH / filename
