from .base import AbstractDataloader
from ..negative_sampler import NegativeSampler
from typing import Optional, Tuple
import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_method

        train_negative_sampler = NegativeSampler(
            train=self.train,
            val=self.val,
            test=self.test,
            user_count=self.user_count,
            item_count=self.item_count,
            sample_size=args.train_negative_sample_size,
            seed=args.train_negative_sampling_seed,
            save_folder=self.save_folder,
        )

        test_negative_sampler = NegativeSampler(
            train=self.train,
            val=self.val,
            test=self.test,
            user_count=self.user_count,
            item_count=self.item_count,
            sample_size=args.test_negative_sample_size,
            seed=args.test_negative_sampling_seed,
            save_folder=self.save_folder,
        )

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(
            dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(
            self.train,
            self.max_len,
            self.mask_prob,
            self.CLOZE_MASK_TOKEN,
            self.item_count,
            self.rng,
        )
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode="val")

    def _get_test_loader(self):
        return self._get_eval_loader(mode="test")

    def _get_eval_loader(self, mode):
        batch_size = (
            self.args.val_batch_size if mode == "val" else self.args.test_batch_size
        )
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == "val" else self.test
        dataset = BertEvalDataset(
            self.train,
            answers,
            self.max_len,
            self.CLOZE_MASK_TOKEN,
            self.test_negative_samples,
        )
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len :]
        labels = labels[-self.max_len :]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]


class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples
        self.u2answer = u2answer

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len :]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return (
            torch.LongTensor(seq),
            torch.LongTensor(candidates),
            torch.LongTensor(labels),
        )


class BertDataset(data_utils.Dataset):
    def __init__(
        self,
        u2seq,
        max_len,
        mask_token,
        eval=False,
        # for train
        num_items=0,
        mask_prob=0,
        rng=None,
        # for eval
        negative_samples=None,
        u2answer=None,
    ):

        if eval:
            if negative_samples is None or u2answer is None:
                raise ValueError("negative_samples and u2answer must be provided")
        if not eval:
            if num_items == 0 or mask_prob == 0 or rng is None:
                raise ValueError("num_items, mask_prob and rng must be provided")
        self.eval = eval
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.negative_samples = negative_samples
        self.u2answer = u2answer

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[index]
        seq = self.u2seq[user]

        if self.eval:
            answer = self.u2answer[user]
            negs = self.negative_samples[user]

            candidates = answer + negs
            labels = [1] * len(answer) + [0] * len(negs)

            seq = seq + [self.mask_token]
            seq = seq[-self.max_len :]
            padding_len = self.max_len - len(seq)
            seq = [0] * padding_len + seq

            return (
                torch.LongTensor(seq),
                torch.LongTensor(candidates),
                torch.LongTensor(labels),
            )

        else:
            tokens = []
            labels = []
            for s in seq:
                prob = self.rng.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(self.rng.randint(1, self.num_items))
                    else:
                        tokens.append(s)

                    labels.append(s)
                else:
                    tokens.append(s)
                    labels.append(0)

            tokens = tokens[-self.max_len :]
            labels = labels[-self.max_len :]

            mask_len = self.max_len - len(tokens)

            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels

            return torch.LongTensor(tokens), torch.LongTensor(labels), torch.empty((0))
