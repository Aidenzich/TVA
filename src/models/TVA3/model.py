import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.feedforward import PositionwiseFeedForward, PointWiseFeedForward
from ...modules.auto_correlation import AutoCorrelation, AutoCorrelationLayer
from ...modules.utils import SublayerConnection
from .utils import rpf1_for_ks


class TVAModel(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        heads: int,
        num_items: int,
        max_len: int,
        dropout: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.max_len = max_len
        self.tva = TVA(
            model_init_seed=0,
            max_len=self.max_len,
            num_items=num_items,
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            dropout=dropout,
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

    def forward(self, batch):
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

    def validation_step(self, batch, batch_idx):
        candidates = batch["candidates"]
        labels = batch["labels"]

        scores = self.forward(batch=batch)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, [1, 10, 20, 30, 50])

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def test_step(self, batch, batch_idx):
        candidates = batch["candidates"]
        labels = batch["labels"]

        scores = self.forward(batch=batch)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, [1, 10, 20, 30, 50])
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))


class TVA(nn.Module):
    def __init__(
        self,
        model_init_seed: int,
        max_len: int,
        num_items: int,
        n_layers: int,
        heads: int,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        # fix_random_seed_as(model_init_seed)
        # self.init_weights()
        vocab_size = num_items + 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = TVAEmbedding(
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
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.time = PositionalEmbedding(max_len=max_len, d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        self.out = nn.Linear(embed_size * 4, embed_size)

        self.latent_emb = nn.Linear(512, embed_size)
        self.latent_emb2 = nn.Linear(512, embed_size)
        self.time_interval = nn.Linear(1, embed_size)
        self.max_len = max_len

        self.tv_emb = nn.Linear(embed_size, embed_size)
        # self.ff = PositionwiseFeedForward(
        #     d_model=embed_size, d_ff=embed_size, dropout=dropout
        # )
        self.ff = PointWiseFeedForward(d_model=embed_size, dropout=dropout)

    def forward(self, batch):
        seqs = batch["item_seq"]
        time_interval_seqs = batch["time_interval_seq"]
        user_latent_factor = batch["userwise_latent_factor"]

        items = self.token(seqs)

        mu = user_latent_factor[:, :512]
        sigma = user_latent_factor[:, 512:]

        mu = mu.unsqueeze(1).repeat(1, self.max_len, 1)
        sigma = sigma.unsqueeze(1).repeat(1, self.max_len, 1)

        positions = self.position(seqs)

        latent = self.latent_emb(mu)
        latent2 = self.latent_emb2(sigma)

        time_interval_seqs = time_interval_seqs.unsqueeze(2)
        time_interval = self.ff(self.time_interval(time_interval_seqs))

        x = self.out(
            torch.cat([items + positions, time_interval, latent, latent2], dim=-1)
        )

        return self.dropout(x)


class TimeEmbedding(nn.Module):
    def __init__(self, max_len, d_model) -> None:
        super().__init__()
        self.te = nn.Embedding(max_len, d_model)

    def forward(self, x) -> Tensor:
        batch_size = x.size(0)
        return self.te.weight.unsqueeze(0).repeat(batch_size, 1, 1)


# Transformer
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, dropout):
        """
        :param d_model: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*d_model
        :param dropout: dropout rate
        """

        super().__init__()
        # self.attention = MultiHeadedAttention(
        #     h=attn_heads, d_model=hidden, dropout=dropout
        # )

        self.attention = AutoCorrelationLayer(
            AutoCorrelation(True, 20, attention_dropout=dropout, output_attention=True),
            d_model,
            attn_heads,
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):

        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )

        x = self.output_sublayer(x, self.feed_forward)

        return self.dropout(x)
