"""
Current this model is not working as paper
"""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict

from src.modules.sasrec_modules import Encoder, LayerNorm, Decoder, Intermediate
from ...modules.utils import SCHEDULER
from ...metrics import recalls_and_ndcgs_for_ks, METRICS_KS


class SASRecModel(pl.LightningModule):
    def __init__(
        self,
        num_items: int,
        trainer_config: Dict,
        model_params: Dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.max_len = model_params["max_len"]
        self.l2_emb = model_params["l2_emb"]

        self.d_model = model_params["d_model"]
        self.max_len = model_params["max_len"]
        
        self.model = SASRec(
            num_items,
            hidden_size=self.d_model,
            max_seq_length=self.max_len,
            num_attention_heads=model_params["heads"],
            attention_probs_dropout_prob=model_params["attention_dropout"],
            hidden_dropout_prob=model_params["dropout"],
            hidden_act=model_params["activation"],
            num_hidden_layers=model_params["n_layers"],
            initializer_range=model_params["initializer_range"],
        )

        self.lr_metric = 0
        self.lr = trainer_config["lr"]
        self.lr_scheduler = SCHEDULER.get(trainer_config["lr_scheduler"], None)
        self.lr_scheduler_args = trainer_config["lr_scheduler_args"]
        self.lr_scheduler_interval = trainer_config.get("lr_scheduler_interval", "step")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.lr_scheduler != None:
            lr_schedulers = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_args),
                "interval": self.lr_scheduler_interval,
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def forward(self, batch):
        item_seq = batch["train_seq"]        

        return self.sasrec(item_seq)

    def training_step(self, batch, batch_idx) -> Tensor:
        sequence_output = self.forward(batch)
        target_pos

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
        if self.lr_scheduler != None:
            sch = self.lr_schedulers()
            sch.step()

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        pass

    def test_step(self, batch, batch_idx) -> None:
        pass


class SASRec(nn.Module):
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
        initializer_range,
    ) -> None:
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.item_encoder = Encoder(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers,
        )
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.intermediate = Intermediate(
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
        )

        # projection on discriminator output
        self.dis_projection = nn.Linear(hidden_size, 1)
        self.initializer_range = initializer_range
        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    # Positional Embedding
    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def forward(self, input_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        if len(item_encoded_layers) == 0:
            sequence_output = self.intermediate(sequence_emb)
        else:
            sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
