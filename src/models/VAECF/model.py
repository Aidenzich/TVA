import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import trange
import pytorch_lightning as pl
from typing import List
from .utils import split_matrix_by_mask, recall_precision_f1_calculate
from ...modules.utils import SCHEDULER


class VAECFModel(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        trainer_config: dict,
        model_params: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = model_params["hidden_dim"]
        self.act_fn = model_params["act_fn"]
        self.autoencoder_structure = model_params["autoencoder_structure"]
        self.likelihood = model_params["likelihood"]
        self.beta = model_params["beta"]
        self.annealing_epoch = model_params["beta_annealing_epoch"]
        self.weight_decay = trainer_config.get("weight_decay", 0.0)

        self.lr = trainer_config.get("lr", 1e-3)
        self.lr_scheduler = SCHEDULER.get(trainer_config.get("lr_scheduler", None))

        if self.lr_scheduler != None:
            print("Using lr_scheduler: ", self.lr_scheduler)
            self.lr_scheduler_args = trainer_config["lr_scheduler_args"]
            self.lr_scheduler_interval = trainer_config.get(
                "lr_scheduler_interval", "step"
            )

        self.vae = VAE(
            z_dim=self.hidden_dim,
            ae_structure=[num_items] + self.autoencoder_structure,
            activation_function=self.act_fn,
            likelihood=self.likelihood,
        )

        self.top_k = 50

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.lr_scheduler != None:
            lr_schedulers = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_args),
                "interval": self.lr_scheduler_interval,
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def training_step(self, batch, batch_idx) -> Tensor:
        _batch, mu, logvar = self.vae(batch)

        # Annealing beta by epoch
        if self.annealing_epoch == 0:
            annealing_beta = self.beta
        else:
            annealing_beta = min(
                self.beta, self.beta * (self.current_epoch / self.annealing_epoch)
            )

        loss = self.vae.loss(batch, _batch, mu, logvar, annealing_beta)

        self.log("train_loss", loss / len(batch), sync_dist=True)
        self.log("beta", annealing_beta, sync_dist=True)
        return loss / len(batch)

    def validation_step(self, batch, batch_idx) -> None:
        x, true_y, _ = split_matrix_by_mask(batch)

        z_u, _ = self.vae.encode(x)
        pred_y = self.vae.decode(z_u)
        seen = x != 0
        pred_y[seen] = 0
        true_y[seen] = 0

        recall, precision, f1 = recall_precision_f1_calculate(
            pred_y, true_y, k=self.top_k
        )

        self.log(f"vae_recall@{self.top_k}", recall, sync_dist=True)

    def test_step(self, batch, batch_idx) -> None:
        x, true_y, _ = split_matrix_by_mask(batch)
        z_u, _ = self.vae.encode(x)
        pred_y = self.vae.decode(z_u)

        seen = x != 0
        pred_y[seen] = 0
        true_y[seen] = 0

        recall, precision, f1 = recall_precision_f1_calculate(
            pred_y, true_y, k=self.top_k
        )

        self.log(f"vae_recall@{self.top_k}", recall, sync_dist=True)


EPS = 1e-10
torch.set_default_dtype(torch.float32)


# Available Activation functions
ACT = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
}


class VAE(nn.Module):
    def __init__(self, z_dim, ae_structure, activation_function, likelihood):
        super(VAE, self).__init__()

        self.likelihood = likelihood
        self.activation_function = ACT.get(activation_function, None)
        if self.activation_function is None:
            raise ValueError("Supported activation_function: {}".format(ACT.keys()))

        # Encoder
        self.encoder = nn.Sequential()
        for i in range(len(ae_structure) - 1):
            self.encoder.add_module(
                "fc{}".format(i), nn.Linear(ae_structure[i], ae_structure[i + 1])
            )
            self.encoder.add_module("act{}".format(i), self.activation_function)
        self.enc_mu = nn.Linear(ae_structure[-1], z_dim)  # mu
        self.enc_logvar = nn.Linear(ae_structure[-1], z_dim)  # logvar

        # Decoder
        ae_structure = [z_dim] + ae_structure[::-1]
        self.decoder = nn.Sequential()
        for i in range(len(ae_structure) - 1):
            self.decoder.add_module(
                "fc{}".format(i), nn.Linear(ae_structure[i], ae_structure[i + 1])
            )
            if i != len(ae_structure) - 2:
                self.decoder.add_module("act{}".format(i), self.activation_function)

    def encode(self, x):
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z):
        h = self.decoder(z)
        if self.likelihood == "mult":
            return torch.softmax(h, dim=1)
        else:
            return torch.sigmoid(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, x_, mu, logvar, beta):
        # Likelihood
        ll_choices = {
            "mult": x * torch.log(x_ + EPS),
            "bern": (x * torch.log(x_ + EPS) + (1 - x) * torch.log(1 - x_ + EPS)),
            "gaus": -((x - x_) ** 2),
            "pois": x * torch.log(x_ + EPS) - x_,
        }

        ll = ll_choices.get(self.likelihood, None)
        if ll is None:
            raise ValueError("Only supported likelihoods: {}".format(ll_choices.keys()))

        ll = torch.sum(ll, dim=1)

        # KL term
        std = torch.exp(0.5 * logvar)
        kld = -0.5 * (1 + 2.0 * torch.log(std) - mu.pow(2) - std.pow(2))
        kld = torch.sum(kld, dim=1)

        return torch.mean(beta * kld - ll)
