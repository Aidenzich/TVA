from site import venv
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from src.configs import RED_COLOR, END_COLOR
from .utils import rpf1_for_ks


# BERTModel
class TVAModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
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
            hidden_size=hidden_size,
            heads=heads,
            dropout=dropout,
        )
        self.out = nn.Linear(hidden_size, num_items + 1)
        self.lr_metric = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        lr_schedulers = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10
            ),
            "monitor": "train_loss",
        }

        return [optimizer], [lr_schedulers]

    def forward(self, x, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor):
        x = self.tva(x, vae_seqs, time_seqs, time_interval_seqs, user_latent_factor)
        return self.out(x)

    def training_step(self, batch, batch_idx):
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
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        # fix_random_seed_as(model_init_seed)
        # self.init_weights()
        vocab_size = num_items + 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = TVAEmbedding(
            vocab_size=vocab_size,
            embed_size=hidden_size,
            max_len=max_len,
            dropout=dropout,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, heads, hidden_size * 4, dropout)
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
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, hidden_units=embed_size)

        # self.time = TimeEmbedding2(max_len=max_len, hidden_units=embed_size)
        self.time = TimeEmbedding(max_len=max_len, hidden_units=embed_size)
        # self.time = TokenEmbedding3(vocab_size=max_len, embed_size=embed_size)

        self.time_interval = TimeEmbedding(max_len=max_len, hidden_units=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        self.out = nn.Linear(embed_size * 4, embed_size)
        # self.pre = nn.Linear(embed_size, embed_size)
        self.latent_emb = nn.Linear(512, embed_size)
        self.max_len = max_len

        # self.tv_emb = nn.Linear(max_len + 512, embed_size)
        # self.new = PositionalEmbedding(max_len=max_len, hidden_units=embed_size)

    def forward(
        self,
        sequence,
        vae_sequence,
        time_sequence,
        time_interval_seqs,
        user_latent_factor,
    ):

        # vae_sequence = F.softmax(vae_sequence, dim=1)

        # vae_x = vae_sequence.unsqueeze(2).repeat(
        #     1, 1, self.embed_size
        # )  # Batch x Seq_len to Batch x Seq_len x Embed_size

        items = self.token(sequence)
        user_latent_factor = user_latent_factor.unsqueeze(1).repeat(1, self.max_len, 1)

        positions = self.position(sequence)

        latent = self.latent_emb(user_latent_factor)

        time = self.time(time_sequence)
        time_interval = self.time_interval(time_interval_seqs)

        # tv = torch.cat([user_latent_factor, time_sequence], dim=-1)
        # tv_emb = self.new(self.tv_emb(tv))
        # print(tv.shape)
        # print(tv_emb.shape)
        # print(items.shape)
        # print(self.new(tv_emb).shape)

        # torch.mul == *
        # torch.bmm(latent, time)
        # item_time = self.tv_emb(torch.matmul(x, time_interval.transpose(-2, -1)))
        # item_latent = torch.matmul(items, latent.transpose(-2, -1))

        # 0.3708 [items, latent, time_interval, time]

        # x = items + time + time_interval  # [12, 128, 256] 12 batch, 128 seq, 256 embed
        x = self.out(torch.cat([items, latent, time_interval, positions], dim=-1))
        # x = self.out(torch.cat([items, latent, time, time_interval], dim=-1))
        return self.dropout(x)


class TimeEmbedding(nn.Module):
    def __init__(self, max_len, hidden_units):
        super().__init__()
        self.te = nn.Embedding(max_len, hidden_units)

    def forward(self, x):
        batch_size = x.size(0)
        return self.te.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TimeEmbedding2(nn.Module):
    def __init__(self, max_len, hidden_units):
        super().__init__()
        self.te = nn.Linear(hidden_units, max_len)

    def forward(self, x):
        batch_size = x.size(0)
        return self.te.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_units):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, hidden_units)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class TransformerTokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
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

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, hidden_units=hidden, dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_units=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        # self.feed_forward = PointWiseFeedForward(hidden_units=hidden, dropout=dropout)

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


# Attention
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, hidden_units, dropout=0.1):
        super().__init__()

        assert hidden_units % h == 0, (
            RED_COLOR + "model size must be divisible by head size" + END_COLOR
        )

        # We assume d_v always equals d_k
        self.d_k = hidden_units // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_units, hidden_units) for _ in range(3)]
        )
        self.output_linear = nn.Linear(hidden_units, hidden_units)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from hidden_units => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


# Feed Forward
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_units, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_units, d_ff)
        self.w_2 = nn.Linear(d_ff, hidden_units)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout=0.1):  # wried, why fusion X 2?
        super(PointWiseFeedForward, self).__init__()
        self.conv_1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.conv_2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.dropout_2 = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        outputs = self.dropout_2(
            self.conv_2(
                self.relu(self.dropout_1(self.conv_1(inputs.transpose(-1, -2))))
            )
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# Gelu
class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


# Layer Normalization
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# Sublayer Connection
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    'connect Feed Forward then Add & Norm' or 'connect Multi-Head Attention then Add & Norm'
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):  # sublayer is a feed foward
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class SublayerConnection2(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection2, self).__init__()
        self.norm = torch.nn.LayerNorm(size, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        pass

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
