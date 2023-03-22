from typing import Any
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.transformer import TransformerBlock
from ...metrics import rpf1_for_ks, METRICS_KS


class CBiTModel(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        heads: int,
        num_items: int,
        max_len: int,
        dropout: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_len = max_len
        self.bert = CBiT(
            max_len=self.max_len,
            num_items=num_items,
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            dropout=dropout,
        )

        self.out = nn.Linear(d_model, num_items + 1)
        self.lr_scheduler_name = "ReduceLROnPlateau"

    def configure_optimizers(self) -> Any:
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

    def forward(self, x) -> torch.Tensor:
        x = self.bert(x)
        return self.out(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        seqs, labels, _ = batch
        logits = self.forward(
            seqs
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
        seqs, candidates, labels = batch
        scores = self.forward(seqs)  # Batch x Seq_len x Vocab_size
        scores = scores[:, -1, :]  # Batch x Vocab_size
        scores[:, 0] = -999.999
        scores[
            :, -1
        ] = -999.999  # pad token and mask token should not appear in the logits outpu
        scores = scores.gather(1, candidates)  # Batch x Candidates
        metrics = rpf1_for_ks(scores, labels, METRICS_KS)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def test_step(self, batch, batch_idx):
        seqs, candidates, labels = batch
        scores = self.forward(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores[:, 0] = -999.999
        scores[
            :, -1
        ] = -999.999  # pad token and mask token should not appear in the logits outpu
        scores = scores.gather(1, candidates)  # B x C
        metrics = rpf1_for_ks(scores, labels, METRICS_KS)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))


# BERT
class CBiT(nn.Module):
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
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        # + self.segment(segment_label)
        return self.dropout(x)


class NTXENTloss(nn.Module):
    def __init__(self, args, device, temperature=1.0):
        super(NTXENTloss, self).__init__()
        self.args = args
        self.temperature = temperature
        self.projection_dim = args.bert_hidden_units
        self.device = device
        self.w1 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(
            self.device
        )
        self.bn1 = nn.BatchNorm1d(self.projection_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.w2 = nn.Linear(self.projection_dim, self.projection_dim, bias=False).to(
            self.device
        )
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False).to(self.device)
        # self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))

    def cosinesim(self, h1, h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)

    def forward(self, h1, h2, calcsim="dot"):
        b = h1.shape[0]
        if self.args.projectionhead:
            z1, z2 = self.project(
                h1.view(b * self.args.bert_max_len, self.args.bert_hidden_units)
            ), self.project(
                h2.view(b * self.args.bert_max_len, self.args.bert_hidden_units)
            )
        else:
            z1, z2 = h1, h2
        z1 = z1.view(b, self.args.bert_max_len * self.args.bert_hidden_units)
        z2 = z2.view(b, self.args.bert_max_len * self.args.bert_hidden_units)
        if calcsim == "dot":
            sim11 = torch.matmul(z1, z1.T) / self.temperature
            sim22 = torch.matmul(z2, z2.T) / self.temperature
            sim12 = torch.matmul(z1, z2.T) / self.temperature
        elif calcsim == "cosine":
            sim11 = self.cosinesim(z1, z1) / self.temperature
            sim22 = self.cosinesim(z2, z2) / self.temperature
            sim12 = self.cosinesim(z1, z2) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float("-inf")
        sim22[..., range(d), range(d)] = float("-inf")
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        ntxentloss = self.criterion(raw_scores, targets)
        return ntxentloss
