from abc import *
from pathlib import Path
import pickle
import string
import numpy as np
from tqdm import trange
from collections import Counter
from config import DATA_PATH


class NegativeSampler(metaclass=ABCMeta):
    def __init__(
        self,
        train,
        val,
        test,
        user_count,
        item_count,
        sample_size,
        seed,
        method="random",
    ):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.method = method

    def _generate_random_negative_samples(self):
        assert self.seed is not None, "Specify seed for random sampling"
        np.random.seed(self.seed)
        negative_samples = {}
        print("Sampling negative items")
        for user in trange(self.user_count):
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
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) + 1
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples

    def _generate_popular_negative_samples(self):
        popular_items = self.items_by_popularity()

        negative_samples = {}
        print("Sampling negative items")
        for user in trange(self.user_count):
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

    def _get_save_path(self):
        folder = DATA_PATH
        filename = "{}-sample_size{}-seed{}.pkl".format(
            self.method, self.sample_size, self.seed
        )

        return folder / filename

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popular_items

    def get_negative_samples(self, method="random"):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print("Negatives samples exist. Loading.")
            negative_samples = pickle.load(savefile_path.open("rb"))
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
