from typing import Any, Dict
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.transformer import TransformerBlock
from ...modules.utils import SCHEDULER
from ...metrics import recalls_and_ndcgs_for_ks, METRICS_KS
from ...models.BERT4Rec.model import BERT


class CBiTModel(pl.LightningModule):
    def __init__(self, num_items, model_params, trainer_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_len = model_params["max_len"]
        self.d_model = model_params["d_model"]

        self.bert = BERT(
            max_len=self.max_len,
            num_items=num_items,
            n_layers=model_params["n_layers"],
            d_model=self.d_model,
            heads=model_params["heads"],
            dropout=model_params["dropout"],
        )

        self.ks = trainer_config.get("ks", METRICS_KS)
        self.out = nn.Linear(self.d_model, num_items + 1)
        self.lr = trainer_config["lr"]
        self.lr_scheduler = (
            SCHEDULER.get(trainer_config["lr_scheduler"], None)
            if trainer_config.get("lr_scheduler", None)
            else None
        )
        self.lr_scheduler_args = trainer_config.get("lr_scheduler_args", None)
        self.lr_scheduler_interval = trainer_config.get("lr_scheduler_interval", "step")
        self.weight_decay = trainer_config["weight_decay"]

        # CBIT
        self.num_positive = model_params["num_positive"]
        self.temperature = model_params["temperature"]
        self.theta = 0
        self.lambda_ = model_params["lambda_"]
        self.alpha = model_params["alpha"]
        self.NTXENTloss = NTXENTloss(
            max_len=self.max_len, d_model=self.d_model, temperature=self.temperature
        )

        self.calcsim = "cosine"

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

    def forward(self, x) -> torch.Tensor:
        x = self.bert(x)

        return self.out(x), x

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, cl_loss = self.calculate_loss(batch)

        if self.lr_scheduler != None:
            sch = self.lr_schedulers()
            sch.step()

        self.log("cl_loss", cl_loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def test_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log("leave1out_" + metric, torch.FloatTensor([metrics[metric]]))

    def calculate_loss(self, batch):
        num_positive = len(batch) // 2
        pairs = []

        main_loss = torch.LongTensor([0]).to(batch[0].device)
        cl_loss = torch.LongTensor([0]).to(batch[0].device)

        for i in range(num_positive):
            seqs = batch[2 * i]
            labels = batch[2 * i + 1]

            logits_k, c_i_k = self.forward(seqs)
            loss_k = F.cross_entropy(
                logits_k.view(-1, logits_k.size(-1)), labels.view(-1)
            )
            main_loss = main_loss + loss_k
            pairs.append(c_i_k)
        for j in range(num_positive):
            for k in range(num_positive):
                if j != k:
                    cl_loss = (
                        self.NTXENTloss(pairs[j], pairs[k], calcsim=self.calcsim)
                        + cl_loss
                    )

        num_main_loss = main_loss.detach().data.item()
        num_cl_loss = cl_loss.detach().data.item()
        theta_hat = num_main_loss / (num_main_loss + self.lambda_ * num_cl_loss)
        self.theta = self.alpha * theta_hat + (1 - self.alpha) * self.theta
        total_loss = main_loss + self.theta * cl_loss
        return total_loss, cl_loss

    def calculate_metrics(self, batch) -> Dict[str, float]:
        batch_item_seq = batch["item_seq"]
        batch_candidates = batch["candidates"]
        batch_labels = batch["labels"]

        scores, _ = self.forward(batch_item_seq)  # Batch x Seq_len x Item_num
        scores = scores[:, -1, :]  # Batch x Item_num
        scores[:, 0] = -999.999
        scores[
            :, -1
        ] = -999.999  # pad token and mask token should not appear in the logits outpu

        scores = scores.gather(1, batch_candidates)  # Batch x Candidates
        metrics = recalls_and_ndcgs_for_ks(scores, batch_labels, METRICS_KS)
        return metrics


class NTXENTloss(nn.Module):
    def __init__(self, max_len, d_model, projection_head=None, temperature=1.0) -> None:
        super(NTXENTloss, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.projection_head = projection_head
        self.temperature = temperature
        self.projection_dim = d_model

        self.w1 = nn.Linear(self.projection_dim, self.projection_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.projection_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(self.projection_dim, self.projection_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(self.projection_dim, affine=False)

        self.criterion = nn.CrossEntropyLoss()

    def project(self, h):
        return self.bn2(self.w2(self.relu(self.bn1(self.w1(h)))))

    def cosinesim(self, h1, h2):
        h = torch.matmul(h1, h2.T)
        h1_norm2 = h1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = h2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)

    def forward(self, h1, h2, calcsim="dot"):
        b = h1.shape[0]
        if self.projection_head:
            z1, z2 = self.project(
                h1.view(b * self.max_len, self.d_model)
            ), self.project(h2.view(b * self.max_len, self.d_model))
        else:
            z1, z2 = h1, h2
        z1 = z1.view(b, self.max_len * self.d_model)
        z2 = z2.view(b, self.max_len * self.d_model)

        if calcsim not in ["dot", "cosine"]:
            raise ValueError("calcsim must be either dot or cosine")
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
