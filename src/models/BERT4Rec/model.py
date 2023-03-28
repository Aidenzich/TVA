from typing import Any, Dict
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.transformer import TransformerBlock
from ...modules.utils import SCHEDULER
from ...metrics import recalls_and_ndcgs_for_ks, METRICS_KS


class BERTModel(pl.LightningModule):
    def __init__(self, num_items, model_params, trainer_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_len = model_params["max_len"]
        self.d_model = model_params["d_model"]

        self.bert = BERT(
            max_len=self.max_len,
            num_items=num_items,
            n_layers=model_params["n_layers"],
            d_model=self.d_model,
            heads=model_params["heads"],
            dropout=model_params["dropout"],
        )

        self.ks = trainer_config.get("ks", METRICS_KS)
        self.out = nn.Linear(self.d_model, num_items + 1)
        self.lr = trainer_config["lr"]
        self.lr_scheduler = SCHEDULER.get(trainer_config["lr_scheduler"], None)
        self.lr_scheduler_args = trainer_config["lr_scheduler_args"]
        self.lr_scheduler_interval = trainer_config.get("lr_scheduler_interval", "step")
        self.weight_decay = trainer_config["weight_decay"]

    def configure_optimizers(self) -> Any:
        if self.weight_decay != 0:
            print("Using weight decay")
            param = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
            )

        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler != None:
            lr_schedulers = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_args),
                "interval": self.lr_scheduler_interval,
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def forward(self, x) -> torch.Tensor:
        x = self.bert(x)
        return self.out(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        batch_item_seq = batch["item_seq"]
        batch_labels = batch["labels"]

        logits = self.forward(batch_item_seq)  # Batch x Seq_len x Item_num
        logits = logits.view(-1, logits.size(-1))  # (Batch * Seq_len) x Item_num
        batch_labels = batch_labels.view(-1)  # Batch x Seq_len

        loss = F.cross_entropy(
            logits, batch_labels, ignore_index=0, label_smoothing=0.05
        )

        if self.lr_scheduler != None:
            sch = self.lr_schedulers()

            # print(batch_idx)
            # if (batch_idx + 1) == 0:
            # sch.step(self.lr_metric)
            sch.step()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def test_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def calculate_metrics(self, batch) -> Dict[str, float]:
        batch_item_seq = batch["item_seq"]
        batch_candidates = batch["candidates"]
        batch_labels = batch["labels"]

        scores = self.forward(batch_item_seq)  # Batch x Seq_len x Item_num
        scores = scores[:, -1, :]  # Batch x Item_num
        scores[:, 0] = -999.999
        scores[
            :, -1
        ] = -999.999  # pad token and mask token should not appear in the logits outpu

        scores = scores.gather(1, batch_candidates)  # Batch x Candidates
        metrics = recalls_and_ndcgs_for_ks(scores, batch_labels, METRICS_KS)
        return metrics


# BERT
class BERT(nn.Module):
    def __init__(
        self,
        max_len: int,
        num_items: int,
        n_layers: int,
        heads: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        # self.init_weights()
        vocab_size = num_items + 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size,
            embed_size=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass


# Embedding
class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)

        return self.dropout(x)
