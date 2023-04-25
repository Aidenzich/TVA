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
    def __init__(self, num_items, model_params, trainer_config, data_class) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_len = model_params["max_len"]
        self.d_model = model_params["d_model"]
        self.label_smoothing = trainer_config.get("label_smoothing", 0.0)
        self.ks = trainer_config.get("ks", METRICS_KS)
        self.lr = trainer_config["lr"]
        self.lr_scheduler = SCHEDULER.get(trainer_config.get("lr_scheduler"), None)
        self.lr_scheduler_args = trainer_config.get("lr_scheduler_args")
        self.lr_scheduler_interval = trainer_config.get("lr_scheduler_interval", "step")
        self.weight_decay = trainer_config["weight_decay"]
        self.num_mask = model_params.get("num_mask", 1)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(
            vocab_size=num_items + 2,
            embed_size=self.d_model,
            max_len=self.max_len,
            dropout=model_params["dropout"],
        )

        self.model = BERT(
            n_layers=model_params["n_layers"],
            d_model=self.d_model,
            heads=model_params["heads"],
            dropout=model_params["dropout"],
            embedding=self.embedding,
        )

        self.out = nn.Linear(self.d_model, num_items + 1)

    def configure_optimizers(self) -> Any:

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler != None:
            lr_schedulers = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_args),
                "interval": self.lr_scheduler_interval,
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def forward(self, item_seq) -> torch.Tensor:
        item_seq = self.model(item_seq)
        return self.out(item_seq)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        main_loss = torch.FloatTensor([0]).to(batch["item_seq_0"].device)
        for idx in range(self.num_mask):
            batch_item_seq = batch[f"item_seq_{idx}"]
            batch_labels = batch[f"labels_{idx}"]

            logits = self.forward(batch_item_seq)  # Batch x Seq_len x Item_num
            logits = logits.view(-1, logits.size(-1))  # (Batch * Seq_len) x Item_num
            batch_labels = batch_labels.view(-1)  # Batch x Seq_len

            loss = F.cross_entropy(
                logits,
                batch_labels,
                ignore_index=0,
                label_smoothing=self.label_smoothing,
            )
            main_loss = main_loss + loss

        if self.lr_scheduler != None:
            sch = self.lr_schedulers()
            sch.step()

        self.log("train_loss", main_loss, sync_dist=True)
        return main_loss

    def validation_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )

    def test_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )

    def calculate_metrics(self, batch) -> Dict[str, float]:
        scores = self.forward(item_seq=batch["item_seq"])  # Batch x Seq_len x Item_num

        scores = scores[:, -1, :]  # Batch x Item_num
        scores[:, 0] = -999.999
        scores[
            :, -1
        ] = -999.999  # pad token and mask token should not appear in the logits outpu

        batch_candidates = batch["candidates"]
        batch_labels = batch["labels"]

        scores = scores.gather(1, batch_candidates)  # Batch x Candidates
        metrics = recalls_and_ndcgs_for_ks(scores, batch_labels, self.ks)
        return metrics


# BERT
class BERT(nn.Module):
    def __init__(
        self,
        n_layers: int,
        heads: int,
        d_model: int,
        dropout: float,
        embedding: nn.Module = None,
    ) -> None:
        super().__init__()

        self.embedding = embedding

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, **kwargs):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, **kwargs)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


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

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)

        return self.dropout(x)
