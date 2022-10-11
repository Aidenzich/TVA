import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange
import pytorch_lightning as pl
from typing import List


class VAECF(pl.LightningModule):
    def __init__(
        self,
        k: int,
        item_dim: int,
        act_fn: str,
        autoencoder_structure: List[int],
        likelihood: str,
        beta: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.k = k
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.autoencoder_structure = autoencoder_structure
        self.beta = beta
        self.vae = VAE(
            z_dim=self.k,
            ae_structur=[item_dim] + self.ae_structure,
            activation_function=self.act_fn,
            likelihood=self.likelihood,
        )

    def training_step(self, batch, batch_idx):
        _batch, mu, logvar = self.vae(batch)
        loss = self.vae.loss(batch, _batch, mu, logvar, self.beta)
        return loss


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
