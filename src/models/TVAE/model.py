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
from ..VAECF.model import VAE


class TVAModel(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        model_params: Dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # BERT
        max_len = model_params["max_len"]
        d_model = model_params["d_model"]
        heads = model_params["heads"]
        dropout = model_params["dropout"]
        n_layers = model_params["n_layers"]

        self.max_len = max_len
        self.lr = model_params["lr"]

        self.vae_beta = model_params["vae_beta"]

        self.tva = TVA(
            max_len=self.max_len,
            num_items=num_items,
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            dropout=dropout,
            vae_beta=model_params["vae_beta"],
            vae_hidden_dim=model_params["vae_hidden_dim"],
            vae_act_fn=model_params["vae_act_fn"],
            vae_likelihood=model_params["vae_likelihood"],
            vae_autoencoder_structure=model_params["vae_autoencoder_structure"],
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

    def training_step(self, batch, batch_idx) -> Tensor:
        labels = batch["labels"]

        user_matrix = batch["user_matrix"]
        _user_matrix, mu, logvar = self.tva.vae(user_matrix)

        vae_loss = self.tva.vae.loss(
            user_matrix, _user_matrix, mu, logvar, self.tva.vae_beta
        )

        zu_u, zu_sigma = self.tva.vae.encode(user_matrix)  # Batch x Hidden_dim

        user_features = torch.cat([zu_u, zu_sigma], dim=1)

        seqs = batch["item_seq"]
        logits = self.tva(
            seqs=seqs,
            user_features=user_features,
        )

        logits = self.tva(
            seqs=seqs,
            user_features=user_features,
        )  # Batch x Sequence_len x Item_nums
        logits = logits.view(-1, logits.size(-1))  # (B * T) x V
        labels = labels.view(-1)  # B * T

        loss = F.cross_entropy(logits, labels, ignore_index=0)

        if self.lr_scheduler_name == "ReduceLROnPlateau":
            sch = self.lr_schedulers()
            if (batch_idx + 1) == 0:
                sch.step(self.lr_metric)

        self.log("train_loss", loss + 0.1 * vae_loss, sync_dist=True)
        return loss + 0.1 * vae_loss

    def validation_step(self, batch, batch_idx) -> None:
        candidates = batch["candidates"]
        labels = batch["labels"]
        seqs = batch["item_seq"]

        user_matrix = batch["user_matrix"]
        zu_u, zu_sigma = self.tva.vae.encode(user_matrix)  # Batch x Hidden_dim
        user_features = torch.cat([zu_u, zu_sigma], dim=1)

        scores = self.tva(
            seqs=seqs, user_features=user_features
        )  # Batch x Sequence_len x Item
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = rpf1_for_ks(scores, labels, METRICS_KS)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )

    def test_step(self, batch, batch_idx) -> None:
        candidates = batch["candidates"]
        labels = batch["labels"]
        seqs = batch["item_seq"]
        user_matrix = batch["user_matrix"]
        zu_u, zu_sigma = self.tva.vae.encode(user_matrix)  # Batch x Hidden_dim
        user_features = torch.cat([zu_u, zu_sigma], dim=1)

        scores = self.tva(
            seqs=seqs, user_features=user_features
        )  # Batch x Sequence_len x Item

        scores = scores[:, -1, :]  # Batch x Item
        scores = scores.gather(1, candidates)  # Batch x Candidates

        metrics = rpf1_for_ks(scores, labels, METRICS_KS)
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )


class TVA(nn.Module):
    def __init__(
        self,
        max_len: int,
        num_items: int,
        n_layers: int,
        heads: int,
        d_model: int,
        dropout: float,
        vae_beta: float,
        vae_hidden_dim: int,
        vae_act_fn: str,
        vae_likelihood: str,
        vae_autoencoder_structure,
    ) -> None:
        super().__init__()

        vocab_size = num_items + 2

        self.out = nn.Linear(d_model, num_items + 1)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = TVAEmbedding(
            vocab_size=vocab_size,
            embed_size=d_model,
            max_len=max_len,
            dropout=dropout,
            user_latent_factor_dim=20,
        )

        # VAE
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_act_fn = vae_act_fn
        self.vae_likelihood = vae_likelihood
        self.vae_autoencoder_structure = vae_autoencoder_structure
        self.vae_beta = vae_beta

        self.vae = VAE(
            z_dim=self.vae_hidden_dim,
            ae_structure=[num_items] + self.vae_autoencoder_structure,
            activation_function=self.vae_act_fn,
            likelihood=self.vae_likelihood,
        )

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, heads, d_model * 4, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, seqs, user_features):
        """
        batch must have:
            - item_seq: (batch_size, max_len)
            - userwise_features: (batch_size, user_latent_factor_dim)
            - itemwise_features: (batch_size, max_len, item_latent_factor_dim)
        """
        mask = (seqs > 0).unsqueeze(1).repeat(1, seqs.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(
            seqs=seqs,
            user_features=user_features,
        )

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return self.out(x)

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

        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)

        self.out = nn.Linear(embed_size * 3, embed_size)

        self.latent_emb = nn.Linear(user_latent_factor_dim * 2, embed_size)
        self.ff = PositionwiseFeedForward(d_model=embed_size, d_ff=128, dropout=dropout)

    def forward(self, seqs, user_features):
        # Should I term seq into user-wise features and item-wise features here?
        items = self.token(seqs)

        # print(user_features.shape)  # Batch x 40

        u_mu = F.softmax(user_features[:, : self.user_latent_factor_dim], dim=1)
        u_sigma = F.softmax(user_features[:, self.user_latent_factor_dim :], dim=1)

        u_mu = u_mu.unsqueeze(1).repeat(1, self.max_len, 1)
        u_sigma = u_sigma.unsqueeze(1).repeat(1, self.max_len, 1)

        positions = self.position(seqs)

        user_latent = self.latent_emb(torch.cat([u_mu, u_sigma], dim=-1))
        user_latent = self.ff(user_latent)

        x = self.out(
            torch.cat(
                [items, positions, user_latent],
                dim=-1,
            )
        )

        return self.dropout(x)