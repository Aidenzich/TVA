from site import venv
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
from .utils import rpf1_for_ks


# BERTModel
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

    def forward(self, x, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor):
        x = self.tva(x, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor)
        return self.out(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        (
            seqs,
            vae_seqs,
            time_seqs,
            time_interval_seqs,
            user_latent_factor,
            labels,
            _,
        ) = batch

        logits = self.forward(
            seqs, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor
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
        (
            seqs,
            vae_seqs,
            time_seqs,
            time_interval_seqs,
            user_latent_factor,
            candidates,
            labels,
        ) = batch
        scores = self.forward(
            seqs, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor
        )  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, [1, 10, 20, 30, 50])

        for metric in metrics.keys():
            # self.log("bert_" + metric, torch.FloatTensor([metrics[metric]]))
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def test_step(self, batch, batch_idx):
        (
            seqs,
            vae_seqs,
            time_seqs,
            time_interval_seqs,
            user_latent_factor,
            candidates,
            labels,
        ) = batch
        scores = self.forward(
            seqs, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor
        )  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, [1, 10, 20, 30, 50])
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))


# BERT
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
    ) -> None:
        super().__init__()

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

    def forward(
        self,
        x,
        vae_seqs,
        time_seqs,
        time_interval_seqs,
        user_latent_factor,
    ):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(
            x, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor
        )

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass


# Embedding
class TVAEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1) -> None:
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)

        # self.time = PositionalEmbedding(max_len=max_len, d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        self.out = nn.Linear(embed_size * 4, embed_size)

        self.latent_emb = nn.Linear(512, embed_size)
        self.latent_emb2 = nn.Linear(512, embed_size)
        self.time_interval = nn.Linear(1, embed_size)
        self.max_len = max_len

        self.tv_emb = nn.Linear(embed_size, embed_size)
        self.ff = PositionwiseFeedForward(
            d_model=embed_size, d_ff=embed_size, dropout=dropout
        )
        # self.new = PositionalEmbedding(max_len=max_len, d_model=embed_size)

    def forward(
        self,
        sequence,
        vae_sequence,
        time_sequence,
        time_interval_seqs,
        user_latent_factor,
    ):

        vae_sequence = F.softmax(vae_sequence, dim=1)
        vae_sequence = vae_sequence.unsqueeze(2).repeat(
            1, 1, self.embed_size
        )  # Batch x Seq_len to Batch x Seq_len x Embed_size
        vae_sequence = self.tv_emb(vae_sequence)
        # vae_x = vae_sequence.unsqueeze(2).repeat(
        #     1, 1, self.embed_size
        # )  # Batch x Seq_len to Batch x Seq_len x Embed_size

        items = self.token(sequence)

        mu = user_latent_factor[:, : (len(user_latent_factor) / 2)]
        mu = mu.unsqueeze(1).repeat(1, self.max_len, 1)
        sigma = user_latent_factor[:, (len(user_latent_factor) / 2) :]
        sigma = sigma.unsqueeze(1).repeat(1, self.max_len, 1)

        positions = self.position(sequence)

        latent = self.latent_emb(mu)
        latent2 = self.latent_emb2(sigma)

        # time = self.time(time_sequence)
        # time_interval_seqs = time_interval_seqs.unsqueeze(2)
        # time_interval = self.ff(self.time_interval(time_interval_seqs))

        # item_time = self.tv_emb(torch.matmul(x, time_interval.transpose(-2, -1)))
        # item_latent = torch.matmul(items, latent.transpose(-2, -1))

        # x = items + time + time_interval  # [12, 128, 256] 12 batch, 128 seq, 256 embed
        x = self.out(torch.cat([items, positions, latent, latent2], dim=-1))

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
