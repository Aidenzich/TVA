from typing import Any, Dict
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from .temp import Encoder, LayerNorm, Decoder, VariationalDropout, NCELoss, priorKL
from ...modules.utils import SCHEDULER
from ...metrics import recalls_and_ndcgs_for_ks, METRICS_KS
import numpy as np
from .utils import (
    recall_at_k,
    ndcg_k,
    get_metric,
    cal_mrr,
    get_user_performance_perpopularity,
    get_item_performance_perpopularity,
)
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class ContrastVAEModel(pl.LightningModule):
    def __init__(self, num_items, model_params, trainer_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_len = model_params["max_len"]
        self.d_model = model_params["d_model"]
        self.variational_dropout = model_params["variational_dropout"]
        self.train_method = model_params["train_method"]
        self.cl_criterion = NCELoss(model_params["temperature"])
        self.anneal_cap = model_params["anneal_cap"]
        self.total_annealing_step = model_params["total_annealing_step"]
        self.latent_clr_weight = model_params["latent_clr_weight"]
        self.store_latent = model_params["store_latent"]

        assert self.train_method in [
            "variational_dropout",
            "latent_contrastive_learning",
            "latent_data_augmentation",
            "VAandDA",
        ], """train_method should be one of [variational_dropout, 
        latent_contrastive_learning, latent_data_augmentation, VAandDA]"""

        if model_params["variational_dropout"]:

            self.model = ContrastVAE_VD(
                num_items,
                hidden_size=self.d_model,
                variational_dropout=self.variational_dropout,
                max_len=self.max_len,
                hidden_dropout=model_params["dropout"],
                num_attention_heads=model_params["heads"],
                attention_probs_dropout_prob=model_params["attention_dropout"],
                hidden_dropout_prob=model_params["dropout"],
                hidden_act=model_params["activation"],
                num_hidden_layers=model_params["n_layers"],
                reparam_dropout_rate=model_params["reparam_dropout_rate"],
                total_annealing_step=model_params["total_annealing_step"],
                initializer_range=model_params["initializer_range"],
                train_method=model_params["train_method"],
            )
        else:
            self.model = ContrastVAE(
                num_items,
                hidden_size=self.d_model,
                max_seq_length=self.max_len,
                num_attention_heads=model_params["heads"],
                attention_probs_dropout_prob=model_params["attention_dropout"],
                hidden_dropout_prob=model_params["dropout"],
                hidden_act=model_params["activation"],
                num_hidden_layers=model_params["n_layers"],
                reparam_dropout_rate=model_params["reparam_dropout_rate"],
                total_annealing_step=model_params["total_annealing_step"],
                initializer_range=model_params["initializer_range"],
                train_method=model_params["train_method"],
            )

        self.ks = trainer_config.get("ks", METRICS_KS)
        self.lr = trainer_config["lr"]
        self.lr_scheduler = (
            SCHEDULER[trainer_config["lr_scheduler"]]
            if trainer_config.get("lr_scheduler", None)
            else None
        )
        self.lr_scheduler_args = (
            trainer_config["lr_scheduler_args"]
            if trainer_config.get("lr_scheduler_args", None)
            else {}
        )
        self.lr_scheduler_interval = trainer_config.get("lr_scheduler_interval", "step")
        self.weight_decay = trainer_config["weight_decay"]

    def configure_optimizers(self) -> Any:
        if self.weight_decay != 0:
            print("Using weight decay")
            param = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
            )

        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler != None:
            lr_schedulers = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_args),
                "interval": self.lr_scheduler_interval,
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # batch = tuple(t.to(self.device) for t in batch)
        # (
        #     _,
        #     input_ids,
        #     target_pos,
        #     target_neg,
        #     _,
        #     aug_input_ids,
        # ) = batch  # input_ids, target_ids: [b,max_Sq]

        input_ids = batch["input_ids"]
        target_pos = batch["target_pos"]
        target_neg = batch["target_neg"]
        aug_input_ids = batch["aug_input_ids"]

        if self.train_method == "variational_dropout":
            # reconstructed_seq1, reconstructed_seq2, mu, log_var, alpha = self.model.forward(input_ids, self.step)  # shape:b*max_Sq*d
            (
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                alpha,
            ) = self.model.forward(input_ids, 0, self.global_step)

            loss, recons_auc = self.loss_fn_VD_latent_clr(
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                target_pos,
                target_neg,
                self.global_step,
                alpha,
            )

        elif self.train_method == "latent_contrastive_learning":
            (
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
            ) = self.model.forward(input_ids, 0, self.global_step)
            loss, recons_auc = self.loss_fn_latent_clr(
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                target_pos,
                target_neg,
                self.global_step,
            )

        elif self.train_method == "latent_data_augmentation":
            (
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
            ) = self.model.forward(input_ids, aug_input_ids, self.step)
            loss, recons_auc = self.loss_fn_latent_clr(
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                target_pos,
                target_neg,
                self.global_step,
            )

        elif self.train_method == "VAandDA":
            (
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                alpha,
            ) = self.model.forward(input_ids, aug_input_ids, self.global_step)
            loss, recons_auc = self.loss_fn_VD_latent_clr(
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                target_pos,
                target_neg,
                self.global_step,
                alpha,
            )

        else:
            reconstructed_seq1, mu, log_var = self.model.forward(
                input_ids, 0, self.global_step
            )  # shape:b*max_Sq*d
            loss, recons_auc = self.loss_fn_vanila(
                reconstructed_seq1,
                mu,
                log_var,
                target_pos,
                target_neg,
                self.global_step,
            )

        if self.lr_scheduler != None:
            sch = self.lr_schedulers()
            sch.step()

        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.calculate_metrics(batch)
        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

        return metrics

    def predict_full(self, seq_out):
        """
        test_item_emb: torch.Size([12103, 256])
        seq_out: torch.Size([12, 256])
        """
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def calculate_metrics(self, batch) -> Dict[str, float]:
        input_ids = batch["input_ids"]
        target_pos = batch["target_pos"]
        labels = batch["labels"]
        aug_input_ids = batch["aug_input_ids"]
        candidates = batch["candidates"]

        if self.train_method == "variational_dropout":
            (
                recommend_reconstruct1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                alpha,
            ) = self.model.forward(input_ids, 0, self.global_step)

        elif self.train_method == "latent_contrastive_learning":
            (
                recommend_reconstruct1,
                recommend_reconstruct2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
            ) = self.model.forward(input_ids, 0, self.global_step)

        elif self.train_method == "latent_data_augmentation":
            (
                recommend_reconstruct1,
                recommend_reconstruct2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
            ) = self.model.forward(input_ids, aug_input_ids, self.global_step)

        elif self.train_method == "VAandDA":
            (
                recommend_reconstruct1,
                _,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
                alpha,
            ) = self.model.forward(input_ids, aug_input_ids, self.step)

        else:  # vanila beta-vae with transformerr
            (
                recommend_reconstruct1,
                mu,
                log_var,
            ) = self.model.forward(input_ids, 0, self.step)

        # recommend_reconstruct1.shape  # Batch x Max_len x D_Model
        # recommend_output.shape  # Batch x D_Model
        recommend_output = recommend_reconstruct1[:, -1, :]

        rating_pred = self.predict_full(recommend_output)
        rating_pred = rating_pred.gather(1, candidates)

        metrics = recalls_and_ndcgs_for_ks(rating_pred, labels, self.ks)
        return metrics

    def kl_anneal_function(self, anneal_cap, step, total_annealing_step):
        """

        :param step: increment by 1 for every  forward-backward step
        :param k: temperature for logistic annealing
        :param x0: pre-fixed parameter control the speed of anealing. total annealing steps
        :return:
        """
        # borrows from https://github.com/timbmg/Sentence-VAE/blob/master/train.py
        return min(anneal_cap, (1.0 * step) / total_annealing_step)

    def loss_fn_vanila(
        self, reconstructed_seq1, mu, log_var, target_pos_seq, target_neg_seq, step
    ):
        """
        compute kl divergence, reconstruction
        :param sequence_reconstructed: b*max_Sq*d
        :param mu: b*d
        :param log_var: b*d
        :param target_pos_seq: b*max_Sq*d
        :param target_neg_seq: b*max_Sq*d
        :return:
        """

        """compute KL divergence"""

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=-1)
        )
        kld_weight = self.kl_anneal_function(
            self.anneal_cap, step, self.total_annealing_step
        )

        """compute reconstruction loss from Trainer"""
        recons_loss1, recons_auc = self.cross_entropy(
            reconstructed_seq1, target_pos_seq, target_neg_seq
        )

        loss = recons_loss1 + kld_weight * kld_loss

        return loss, recons_auc

    def loss_fn_latent_clr(
        self,
        reconstructed_seq1,
        reconstructed_seq2,
        mu1,
        mu2,
        log_var1,
        log_var2,
        z1,
        z2,
        target_pos_seq,
        target_neg_seq,
        step,
    ):
        """
        compute kl divergence, reconstruction loss and contrastive loss
        :param sequence_reconstructed: b*max_Sq*d
        :param mu: b*d
        :param log_var: b*d
        :param target_pos_seq: b*max_Sq*d
        :param target_neg_seq: b*max_Sq*d
        :return:
        """

        """compute KL divergence"""

        kld_loss1 = torch.mean(
            -0.5 * torch.sum(1 + log_var1 - mu1**2 - log_var1.exp(), dim=-1)
        )
        kld_loss2 = torch.mean(
            -0.5 * torch.sum(1 + log_var2 - mu2**2 - log_var2.exp(), dim=-1)
        )
        kld_weight = self.kl_anneal_function(
            self.anneal_cap, step, self.total_annealing_step
        )

        """compute reconstruction loss from Trainer"""
        recons_loss1, recons_auc = self.cross_entropy(
            reconstructed_seq1, target_pos_seq, target_neg_seq
        )
        recons_loss2, recons_auc = self.cross_entropy(
            reconstructed_seq2, target_pos_seq, target_neg_seq
        )

        """compute clr loss"""
        user_representation1 = torch.sum(z1, dim=1)
        user_representation2 = torch.sum(z2, dim=1)

        contrastive_loss = self.cl_criterion(user_representation1, user_representation2)

        loss = (
            recons_loss1
            + recons_loss2
            + kld_weight * (kld_loss1 + kld_loss2)
            + self.latent_clr_weight * contrastive_loss
        )
        return loss, recons_auc

    def loss_fn_VD_latent_clr(
        self,
        reconstructed_seq1,
        reconstructed_seq2,
        mu1,
        mu2,
        log_var1,
        log_var2,
        z1,
        z2,
        target_pos_seq,
        target_neg_seq,
        step,
        alpha,
    ):
        """
        compute kl divergence, reconstruction loss and contrastive loss
        :param sequence_reconstructed: b*max_Sq*d
        :param mu: b*d
        :param log_var: b*d
        :param target_pos_seq: b*max_Sq*d
        :param target_neg_seq: b*max_Sq*d
        :return:
        """

        """compute KL divergence"""

        kld_loss1 = torch.mean(
            -0.5 * torch.sum(1 + log_var1 - mu1**2 - log_var1.exp(), dim=-1)
        )
        kld_loss2 = torch.mean(
            -0.5 * torch.sum(1 + log_var2 - mu2**2 - log_var2.exp(), dim=-1)
        )
        kld_weight = self.kl_anneal_function(
            self.anneal_cap, step, self.total_annealing_step
        )

        """compute reconstruction loss from Trainer"""
        recons_loss1, recons_auc = self.cross_entropy(
            reconstructed_seq1, target_pos_seq, target_neg_seq
        )
        recons_loss2, recons_auc = self.cross_entropy(
            reconstructed_seq2, target_pos_seq, target_neg_seq
        )

        """compute clr loss"""

        user_representation1 = torch.sum(z1, dim=1)
        user_representation2 = torch.sum(z2, dim=1)
        contrastive_loss = self.cl_criterion(user_representation1, user_representation2)

        """compute priorKL loss"""
        adaptive_alpha_loss = priorKL(alpha)
        loss = (
            recons_loss1
            + recons_loss2
            + kld_weight * (kld_loss1 + kld_loss2)
            + self.latent_clr_weight * contrastive_loss
            + adaptive_alpha_loss
        )

        return loss, recons_auc

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.d_model)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (
            (pos_ids > 0).view(pos_ids.size(0) * self.max_len).float()
        )  # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc


class ContrastVAE(nn.Module):
    def __init__(
        self,
        num_items,
        hidden_size,
        max_seq_length,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        hidden_act,
        num_hidden_layers,
        reparam_dropout_rate,
        total_annealing_step,
        initializer_range,
        train_method,
    ):
        super(ContrastVAE, self).__init__()

        self.train_method = train_method

        self.total_annealing_step = total_annealing_step
        self.initializer_range = initializer_range

        self.item_embeddings = nn.Embedding(num_items + 2, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.item_encoder_mu = Encoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
        )
        self.item_encoder_logvar = Encoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
        )
        self.item_decoder = Decoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
        )
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.latent_dropout = nn.Dropout(reparam_dropout_rate)
        self.apply(self.init_weights)
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)  # shape: b*max_Sq*d
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb  # shape: b*max_Sq*d

    def extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()  # used for mu, var
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64 b*1*1*max_Sq

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(
            torch.ones(attn_shape), diagonal=1
        )  # torch.uint8 for causality
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)  # 1*1*max_Sq*max_Sq
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.to(extended_attention_mask.device)

        # if self.cuda_condition:
        #     subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = (
            extended_attention_mask * subsequent_mask
        )  # shape: b*1*max_Sq*max_Sq
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def eps_anneal_function(self, step):

        return min(1.0, (1.0 * step) / self.total_annealing_step)

    def reparameterization(self, mu, logvar, step):  # vanila reparam

        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else:
            res = mu + std
        return res

    def reparameterization1(self, mu, logvar, step):  # reparam without noise
        std = torch.exp(0.5 * logvar)
        return mu + std

    def reparameterization2(self, mu, logvar, step):  # use dropout

        if self.training:
            std = self.latent_dropout(torch.exp(0.5 * logvar))
        else:
            std = torch.exp(0.5 * logvar)
        res = mu + std
        return res

    def reparameterization3(
        self, mu, logvar, step
    ):  # apply classical dropout on whole result
        std = torch.exp(0.5 * logvar)
        res = self.latent_dropout(mu + std)
        return res

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, sequence_emb, extended_attention_mask):  # forward

        item_encoded_mu_layers = self.item_encoder_mu(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )

        item_encoded_logvar_layers = self.item_encoder_logvar(
            sequence_emb, extended_attention_mask, True
        )

        return item_encoded_mu_layers[-1], item_encoded_logvar_layers[-1]

    def decode(self, z, extended_attention_mask):
        item_decoder_layers = self.item_decoder(
            z, extended_attention_mask, output_all_encoded_layers=True
        )
        sequence_output = item_decoder_layers[-1]
        return sequence_output

    def forward(self, input_ids, aug_input_ids, step):

        sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
        extended_attention_mask = self.extended_attention_mask(input_ids)

        if self.train_method == "latent_contrastive_learning":
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            return (
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
            )

        elif self.train_method == "latent_data_augmentation":
            aug_sequence_emb = self.add_position_embedding(
                aug_input_ids
            )  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(aug_input_ids)

            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            return (
                reconstructed_seq1,
                reconstructed_seq2,
                mu1,
                mu2,
                log_var1,
                log_var2,
                z1,
                z2,
            )

        else:  # vanilla attentive VAE
            mu, log_var = self.encode(sequence_emb, extended_attention_mask)
            z = self.reparameterization(mu, log_var, step)
            reconstructed_seq1 = self.decode(z, extended_attention_mask)
            return reconstructed_seq1, mu, log_var


class ContrastVAE_VD(ContrastVAE):
    def __init__(
        self,
        num_items,
        hidden_size,
        max_len,
        hidden_dropout,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        hidden_act,
        num_hidden_layers,
        total_annealing_step,
        initializer_range,
        train_method,
        variational_dropout,
        reparam_dropout_rate=0.1,
    ) -> None:
        super(ContrastVAE, self).__init__(
            total_annealing_step=total_annealing_step,
            initializer_range=initializer_range,
            train_method=train_method,
            variational_dropout=variational_dropout,
            reparam_dropout_rate=reparam_dropout_rate,
        )

        self.item_embeddings = nn.Embedding(num_items + 2, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)

        self.item_encoder_mu = Encoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
        )
        self.item_encoder_logvar = Encoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
        )
        self.item_decoder = Decoder(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
        )

        self.dropout = nn.Dropout(hidden_dropout)

        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.latent_dropout_VD = VariationalDropout(
            inputshape=[max_len, hidden_size], adaptive="layerwise"
        )
        self.latent_dropout = nn.Dropout(self.reparam_dropout_rate)

        self.apply(self.init_weights)

        self.drop_rate = nn.Parameter(torch.tensor(0.2), requires_grad=True)

    def reparameterization3(self, mu, logvar, step):  # use drop out

        std, alpha = self.latent_dropout_VD(torch.exp(0.5 * logvar))
        res = mu + std
        return res, alpha

    def forward(self, input_ids, augmented_input_ids, step):

        if self.variational_dropout:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)

        else:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            aug_sequence_emb = self.add_position_embedding(
                augmented_input_ids
            )  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(
                augmented_input_ids
            )

            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)

        return (
            reconstructed_seq1,
            reconstructed_seq2,
            mu1,
            mu2,
            log_var1,
            log_var2,
            z1,
            z2,
            alpha,
        )
