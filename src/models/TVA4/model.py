import pytorch_lightning as pl
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.configs import RED_COLOR, END_COLOR
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.feedforward import PositionwiseFeedForward, PointWiseFeedForward
from ...modules.transformer import TransformerBlock
from ...modules.utils import SCHEDULER
from ...metrics import recalls_and_ndcgs_for_ks, METRICS_KS


class TVAModel(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        model_params: Dict,
        trainer_config: Dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_len = model_params["max_len"]
        self.d_model = model_params["d_model"]

        user_latent_factor_params = model_params.get("user_latent_factor", None)
        item_latent_factor_params = model_params.get("item_latent_factor", None)

        assert user_latent_factor_params is not None, (
            RED_COLOR + "user_latent_factor_params is None" + END_COLOR
        )

        self.label_smoothing = trainer_config.get("label_smoothing", 0.0)
        self.max_len = model_params["max_len"]
        self.ks = trainer_config.get("ks", METRICS_KS)

        use_userwise_var = user_latent_factor_params.get("available", False)
        use_itemwise_var = item_latent_factor_params.get("available", False)

        if use_userwise_var:
            print("Using userwise latent factor")
        if use_itemwise_var:
            print("Using itemwise latent factor")

        userwise_var_dim = user_latent_factor_params.get("hidden_dim", 0)
        itemwise_var_dim = item_latent_factor_params.get("hidden_dim", 0)

        self.tva = TVA(
            max_len=self.max_len,
            num_items=num_items,
            n_layers=model_params["n_layers"],
            d_model=self.d_model,
            heads=model_params["heads"],
            dropout=model_params["dropout"],
            user_latent_factor_dim=userwise_var_dim if use_userwise_var else 0,
            item_latent_factor_dim=itemwise_var_dim if use_itemwise_var else 0,
        )
        self.out = nn.Linear(self.d_model, num_items + 1)
        self.lr = (
            trainer_config["lr"] if trainer_config.get("lr", None) is not None else 1e-4
        )
        self.lr_scheduler = SCHEDULER.get(trainer_config["lr_scheduler"], None)
        self.lr_scheduler_args = trainer_config["lr_scheduler_args"]
        self.lr_scheduler_interval = trainer_config.get("lr_scheduler_interval", "step")

        self.weight_decay = trainer_config.get("weight_decay", 0.0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler != None:
            lr_schedulers = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_args),
                "interval": self.lr_scheduler_interval,
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def forward(self, batch) -> Tensor:
        item_seq = batch["item_seq"]
        userwise_latent_factor = batch["userwise_latent_factor"]
        itemwise_latent_factor_seq = batch["itemwise_latent_factor_seq"]

        x = self.tva(item_seq, userwise_latent_factor, itemwise_latent_factor_seq)
        return self.out(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        batch_labels = batch["labels"]
        logits = self.forward(
            batch=batch
        )  # B x T x V (128 x 100 x 3707) (BATCH x SEQENCE_LEN x ITEM_NUM)

        logits = logits.view(-1, logits.size(-1))  # (B * T) x V
        batch_labels = batch_labels.view(-1)  # B * T
        loss = F.cross_entropy(
            logits, batch_labels, ignore_index=0, label_smoothing=self.label_smoothing
        )

        if self.lr_scheduler != None:
            sch = self.lr_schedulers()
            sch.step()

        self.log("train_loss", loss, sync_dist=True)
        return loss

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

        batch_candidates = batch["candidates"]
        batch_labels = batch["labels"]

        scores = self.forward(batch)  # Batch x Seq_len x Item_num
        scores = scores[:, -1, :]  # Batch x Item_num
        scores[:, 0] = -999.999
        scores[
            :, -1
        ] = -999.999  # pad token and mask token should not appear in the logits outpu

        scores = scores.gather(1, batch_candidates)  # Batch x Candidates
        metrics = recalls_and_ndcgs_for_ks(scores, batch_labels, self.ks)
        return metrics


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

    def forward(self, item_seq, userwise_latent_factor, itemwise_latent_factor_seq):
        mask = (item_seq > 0).unsqueeze(1).repeat(1, item_seq.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(item_seq, userwise_latent_factor, itemwise_latent_factor_seq)

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
        user_latent_factor_dim=None,
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

        in_dim = embed_size

        if self.user_latent_factor_dim != 0:
            self.user_latent_emb = nn.Linear(user_latent_factor_dim * 2, embed_size)
            self.user_latent_emb_ff = PositionwiseFeedForward(
                d_model=embed_size, d_ff=8, dropout=dropout
            )
            in_dim += embed_size

        if self.item_latent_factor_dim != 0:
            self.item_latent_emb = nn.Linear(item_latent_factor_dim * 2, embed_size)
            self.item_latent_emb_ff = PositionwiseFeedForward(
                d_model=embed_size, d_ff=32, dropout=dropout
            )
            in_dim += embed_size

        self.out = nn.Linear(in_dim, embed_size)

    def forward(self, item_seq, userwise_latent_factor, itemwise_latent_factor_seq):
        items = self.token(item_seq)
        positions = self.position(item_seq)
        _cat = [items + positions]

        if self.user_latent_factor_dim != 0:
            assert userwise_latent_factor.shape[1] == self.user_latent_factor_dim * 2, (
                RED_COLOR
                + "user latent factor dim is not correct, please check model config"
                + END_COLOR
            )

            u_mu = F.softmax(
                userwise_latent_factor[:, : self.user_latent_factor_dim], dim=1
            )
            u_sigma = F.softmax(
                userwise_latent_factor[:, self.user_latent_factor_dim :], dim=1
            )

            u_mu = u_mu.unsqueeze(1).repeat(1, self.max_len, 1)
            u_sigma = u_sigma.unsqueeze(1).repeat(1, self.max_len, 1)
            user_latent = u_mu + u_sigma
            user_latent = self.user_latent_emb_ff(user_latent)
            _cat.append(user_latent)

        if self.item_latent_factor_dim != 0:
            assert (
                itemwise_latent_factor_seq.shape[2] == self.item_latent_factor_dim * 2
            ), (
                RED_COLOR
                + "item latent factor dim is not match, please check model config"
                + END_COLOR
            )

            item_latent = self.item_latent_emb(itemwise_latent_factor_seq)
            item_latent = self.item_latent_emb_ff(item_latent)
            _cat.append(item_latent)

        x = self.out(
            torch.cat(
                _cat,
                dim=-1,
            )
        )

        return self.dropout(x)
