from typing import Dict
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import torch
import math
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.feedforward import PositionwiseFeedForward, PointWiseFeedForward
from ...modules.attetion import MultiHeadedAttention
from ...modules.utils import SublayerConnection
from src.configs import RED_COLOR, END_COLOR
from ...metrics import rpf1_for_ks


# BERTModel
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
        metrics = rpf1_for_ks(scores, labels, [1, 10, 20, 30, 50])

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def test_step(self, batch, batch_idx) -> None:
        candidates = batch["candidates"]
        labels = batch["labels"]

        scores = self.forward(batch=batch)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, [1, 10, 20, 30, 50])
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))


# TVA Module
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

    def init_weights(self) -> None:
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
        self.embed_size = embed_size
        self.max_len = max_len

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)

        self.out = nn.Linear(embed_size * 4, embed_size)
        self.user_latent_factor_dim = user_latent_factor_dim
        self.latent_emb = nn.Linear(user_latent_factor_dim * 2, embed_size)
        self.time_interval = nn.Linear(1, embed_size)

        self.ff = PositionwiseFeedForward(d_model=embed_size, d_ff=128, dropout=dropout)
        self.time_ff = PointWiseFeedForward(d_model=embed_size, dropout=dropout)

        # self.latent_emb2 = nn.Linear(512, embed_size)
        # self.tv_emb = nn.Linear(embed_size, embed_size)
        # self.time = PositionalEmbedding(max_len=max_len, d_model=embed_size)

    def forward(
        self,
        batch,
    ):

        seqs = batch["item_seq"]
        time_interval_seqs = batch["time_interval_seq"]
        user_latent_factor = batch["userwise_latent_factor"]

        items = self.token(seqs)

        u_mu = F.softmax(user_latent_factor[:, : self.user_latent_factor_dim], dim=1)
        u_sigma = F.softmax(user_latent_factor[:, self.user_latent_factor_dim :], dim=1)

        u_mu = u_mu.unsqueeze(1).repeat(1, self.max_len, 1)
        u_sigma = u_sigma.unsqueeze(1).repeat(1, self.max_len, 1)

        positions = self.position(seqs)

        latent3 = self.latent_emb(torch.cat([u_mu, u_sigma], dim=-1))
        latent3 = self.ff(latent3)

        # time = self.time(time_sequence)
        time_interval_seqs = time_interval_seqs.unsqueeze(2)
        time_interval = self.time_ff(self.time_interval(time_interval_seqs))

        # item_time = self.tv_emb(torch.matmul(x, time_interval.transpose(-2, -1)))
        # item_latent = torch.matmul(items, latent.transpose(-2, -1))

        # x = items + time + time_interval  # [12, 128, 256] 12 batch, 128 seq, 256 embed
        x = self.out(torch.cat([items, positions, time_interval, latent3], dim=-1))

        return self.dropout(x)


class TransformerTokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512) -> None:
        # super().__init__(vocab_size, embed_size, padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.emb_size = embed_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Transformer
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, dropout) -> None:
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*d_model
        :param dropout: dropout rate
        """

        super().__init__()

        assert d_model % attn_heads == 0, (
            RED_COLOR + "model size must be divisible by head size" + END_COLOR
        )

        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=d_model, dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=feed_forward_hidden, dropout=dropout
        )
        # self.feed_forward = PointWiseFeedForward(d_model=hidden, dropout=dropout)

        self.input_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
