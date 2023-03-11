import pytorch_lightning as pl
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.configs import RED_COLOR, END_COLOR
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.feedforward import PositionwiseFeedForward, PointWiseFeedForward
from ...modules.transformer import TransformerBlock
from ...metrics import rpf1_for_ks, METRICS_KS


class TVAModel(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        model_params: Dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        max_len = model_params["max_len"]
        d_model = model_params["d_model"]
        heads = model_params["heads"]
        dropout = model_params["dropout"]
        n_layers = model_params["n_layers"]
        user_latent_factor_params = model_params.get("user_latent_factor", None)
        item_latent_factor_params = model_params.get("item_latent_factor", None)

        assert user_latent_factor_params is not None, (
            RED_COLOR + "user_latent_factor_params is None" + END_COLOR
        )

        self.max_len = max_len
        self.tva = TVA(
            max_len=self.max_len,
            num_items=num_items,
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            dropout=dropout,
            user_latent_factor_dim=user_latent_factor_params.get("hidden_dim"),
            item_latent_factor_dim=item_latent_factor_params.get("hidden_dim"),
        )
        self.out = nn.Linear(d_model, num_items + 1)
        self.lr_metric = 0
        self.lr_scheduler_name = "ReduceLROnPlateau"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        if self.lr_scheduler_name == "ReduceLROnPlateau":
            lr_schedulers = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=10
                ),
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def forward(self, batch) -> Tensor:
        x = self.tva(batch=batch)
        return self.out(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        labels = batch["labels"]

        logits = self.forward(
            batch=batch
        )  # B x T x V (128 x 100 x 3707) (BATCH x SEQENCE_LEN x ITEM_NUM)

        logits = logits.view(-1, logits.size(-1))  # (B * T) x V
        labels = labels.view(-1)  # B * T
        loss = F.cross_entropy(logits, labels, ignore_index=0)

        if self.lr_scheduler_name == "ReduceLROnPlateau":
            sch = self.lr_schedulers()
            if (batch_idx + 1) == 0:
                sch.step(self.lr_metric)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        candidates = batch["candidates"]
        labels = batch["labels"]

        scores = self.forward(batch=batch)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, METRICS_KS)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def test_step(self, batch, batch_idx) -> None:
        candidates = batch["candidates"]
        labels = batch["labels"]

        scores = self.forward(batch=batch)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, METRICS_KS)
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))


class TVA(nn.Module):
    def __init__(
        self,
        max_len: int,
        num_items: int,
        n_layers: int,
        heads: int,
        d_model: int,
        dropout: float,
        user_latent_factor_dim: int,
        item_latent_factor_dim: int,
    ) -> None:
        super().__init__()

        vocab_size = num_items + 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = TVAEmbedding(
            vocab_size=vocab_size,
            embed_size=d_model,
            max_len=max_len,
            dropout=dropout,
            user_latent_factor_dim=user_latent_factor_dim,
            item_latent_factor_dim=item_latent_factor_dim,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, batch):
        seqs = batch["item_seq"]
        mask = (seqs > 0).unsqueeze(1).repeat(1, seqs.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(batch=batch)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass


# Embedding
class TVAEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        max_len,
        user_latent_factor_dim,
        item_latent_factor_dim=None,
        dropout=0.1,
    ) -> None:
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()

        # parameters
        self.embed_size = embed_size
        self.max_len = max_len
        self.user_latent_factor_dim = user_latent_factor_dim
        self.item_latent_factor_dim = item_latent_factor_dim

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)

        self.out = nn.Linear(embed_size * 3, embed_size)

        self.latent_emb = nn.Linear(user_latent_factor_dim * 2, embed_size)
        self.time_interval = nn.Linear(1, embed_size)

        self.item_latent_emb = nn.Linear(item_latent_factor_dim * 2, embed_size)

        self.ff = PositionwiseFeedForward(d_model=embed_size, d_ff=128, dropout=dropout)
        self.time_ff = PointWiseFeedForward(d_model=embed_size, dropout=dropout)
        self.item_latent_emb_ff = PositionwiseFeedForward(
            d_model=embed_size, d_ff=128, dropout=dropout
        )

        self.interval_sigmoid = nn.Sigmoid()

    def forward(self, batch):
        seqs = batch["item_seq"]
        time_interval_seqs = batch["time_interval_seq"]
        user_latent_factor = batch["userwise_latent_factor"]
        item_latent_factor_seq = batch["itemwise_latent_factor_seq"]

        items = self.token(seqs)

        assert user_latent_factor.shape[1] == self.user_latent_factor_dim * 2, (
            RED_COLOR
            + "user latent factor dim is not correct, please check model config"
            + END_COLOR
        )

        assert item_latent_factor_seq.shape[2] == self.item_latent_factor_dim * 2, (
            RED_COLOR
            + "item latent factor dim is not match, please check model config"
            + END_COLOR
        )

        u_mu = F.softmax(user_latent_factor[:, : self.user_latent_factor_dim], dim=1)
        u_sigma = F.softmax(user_latent_factor[:, self.user_latent_factor_dim :], dim=1)

        u_mu = u_mu.unsqueeze(1).repeat(1, self.max_len, 1)
        u_sigma = u_sigma.unsqueeze(1).repeat(1, self.max_len, 1)

        positions = self.position(seqs)

        user_latent = self.latent_emb(torch.cat([u_mu, u_sigma], dim=-1))
        user_latent = self.ff(user_latent)

        # time_interval_seqs = F.softmax(time_interval_seqs, dim=1)
        # time_interval_seqs = time_interval_seqs.unsqueeze(2)
        # time_interval_seqs = self.time_interval(time_interval_seqs)

        item_latent = self.item_latent_emb(item_latent_factor_seq)

        x = self.out(
            torch.cat(
                [
                    items,
                    positions,
                    # user_latent,
                    item_latent,
                ],
                dim=-1,
            )
        )

        return self.dropout(x)


class TimeEmbedding(nn.Module):
    def __init__(self, max_len, d_model) -> None:
        super().__init__()
        self.te = nn.Embedding(max_len, d_model)

    def forward(self, x) -> Tensor:
        batch_size = x.size(0)
        return self.te.weight.unsqueeze(0).repeat(batch_size, 1, 1)
