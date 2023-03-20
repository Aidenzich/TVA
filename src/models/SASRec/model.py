import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict

from src.modules.feedforward import PointWiseFeedForward
from ...metrics import rpf1_for_ks, METRICS_KS


class SASRecModel(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        num_users: int,
        model_params: Dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.max_len = model_params["max_len"]
        self.l2_emb = model_params["l2_emb"]
        self.lr = model_params["lr"]

        d_model = model_params["d_model"]
        heads = model_params["heads"]
        dropout = model_params["dropout"]
        n_layers = model_params["n_layers"]

        self.sasrec = SASRec(
            user_num=num_users,
            item_num=num_items,
            max_len=self.max_len,
            d_model=d_model,
            heads=heads,
            dropout=dropout,
            n_layers=n_layers,
        )

        self.lr_metric = 0
        self.lr_scheduler_name = "ReduceLROnPlateau"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler_name == "ReduceLROnPlateau":
            lr_schedulers = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=10
                ),
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        if self.lr_scheduler_name == "LambdaLR":
            lr_schedulers = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda epoch: 0.95**epoch
                ),
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        if self.lr_scheduler_name == "StepLR":
            lr_schedulers = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=10, gamma=0.1
                ),
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def forward(self, batch):
        log_seqs = batch["train_seq"]
        pos_seqs = batch["pos_label"]
        neg_seqs = batch["neg_label"]

        return self.sasrec(log_seqs=log_seqs, pos_seqs=pos_seqs, neg_seqs=neg_seqs)

    def training_step(self, batch, batch_idx) -> Tensor:
        pos_logits, neg_logits = self.forward(batch=batch)

        pos_labels = torch.ones(pos_logits.shape).to(pos_logits.device)
        neg_labels = torch.zeros(neg_logits.shape).to(pos_logits.device)

        indices = np.where(batch["pos_label"].cpu() != 0)
        loss = F.binary_cross_entropy_with_logits(
            pos_logits[indices], pos_labels[indices]
        )
        loss += F.binary_cross_entropy_with_logits(
            neg_logits[indices], neg_labels[indices]
        )

        for param in self.sasrec.item_emb.parameters():
            loss += self.l2_emb * torch.norm(param)

        # Step the lr scheduler
        if self.lr_scheduler_name == "ReduceLROnPlateau":
            sch = self.lr_schedulers()
            if (batch_idx + 1) == 0:
                sch.step(self.lr_metric)

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        candidates = batch["candidates"]
        log_seqs = batch["train_seq"]
        labels = batch["labels"]

        logits = -self.sasrec.predict(log_seqs, candidates)  # Batch x Candidates

        metrics = rpf1_for_ks(logits, labels, METRICS_KS)
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )

    def test_step(self, batch, batch_idx) -> None:
        candidates = batch["candidates"]
        log_seqs = batch["train_seq"]
        labels = batch["labels"]

        logits = self.sasrec.predict(log_seqs, candidates)
        metrics = rpf1_for_ks(logits, labels, METRICS_KS)
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )


class SASRec(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        max_len: int,
        d_model: int,
        heads: int,
        dropout: float,
        n_layers: int,
    ) -> None:
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num

        self.item_emb = nn.Embedding(self.item_num + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(p=dropout)

        self.attention_layernorms = nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(d_model, eps=1e-8)

        for _ in range(n_layers):
            new_attn_layernorm = nn.LayerNorm(d_model, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(d_model, heads, dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(d_model, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(d_model, dropout)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):

        seqs = self.item_emb(log_seqs)

        seqs *= self.item_emb.embedding_dim**0.5

        positions = torch.LongTensor(
            np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        ).to(log_seqs.device)

        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == 0
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool)).to(
            log_seqs.device
        )

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            # key_padding_mask=timeline_mask
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # Batch x Sequence x d_model

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices)  # Batch x Candidates x d_model

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits  # Batch x Candidates
