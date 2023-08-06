from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ...modules.sasrec_modules import Encoder, LayerNorm, Intermediate
from ...modules.utils import SCHEDULER
from ...metrics import recalls_and_ndcgs_for_ks, METRICS_KS
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding


class SASRecModel(pl.LightningModule):
    def __init__(self, num_items, model_params, trainer_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.max_len = model_params["max_len"]
        self.d_model = model_params["d_model"]
        self.label_smoothing = trainer_config.get("label_smoothing", 0.0)
        self.model = SASRec(
            num_items=num_items,
            hidden_size=self.d_model,
            max_seq_length=self.max_len,
            num_attention_heads=model_params["heads"],
            attention_probs_dropout_prob=model_params["attention_dropout"],
            hidden_dropout_prob=model_params["dropout"],
            hidden_act=model_params["activation"],
            num_hidden_layers=model_params["n_layers"],
            initializer_range=model_params["initializer_range"],
        )

        self.loss_type = model_params.get("loss_type", "vallina")
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

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        input_ids = batch["input_ids"]
        target_pos = batch["target_pos"]
        target_neg = batch["target_neg"]

        seq_out = self.model.forward(input_ids)

        if self.loss_type == "bce":
            loss = self.binary_cross_entropy_loss(seq_out, target_pos, target_neg)

        if self.loss_type == "ce":
            loss = self.cross_entropy_loss(seq_out, target_pos)

        self.log("train_loss", loss)

        return loss

    def binary_cross_entropy_loss(self, seq_out, pos_ids, neg_ids):
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

        return loss

    def cross_entropy_loss(self, seq_out, pos_ids):
        seq_emb = seq_out.view(-1, self.d_model)
        test_item_emb = self.model.item_embeddings.weight
        logits = torch.matmul(seq_emb, test_item_emb.transpose(0, 1))

        loss = F.cross_entropy(
            logits,
            torch.squeeze(pos_ids.view(-1)),
            label_smoothing=self.label_smoothing,
        )
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

    def calculate_metrics(self, batch):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        candidates = batch["candidates"]

        rec_output = self.model.forward(input_ids)
        rec_output = rec_output[:, -1, :]
        rating_pred = self.predict_full(rec_output)
        rating_pred = rating_pred.gather(1, candidates)

        metrics = recalls_and_ndcgs_for_ks(rating_pred, labels, self.ks)
        return metrics

    def predict_full(self, seq_out):
        """
        test_item_emb: torch.Size([12103, 256])
        seq_out: torch.Size([12, 256])
        """
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
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

        self.item_embeddings = TokenEmbedding(
            vocab_size=num_items + 2, embed_size=hidden_size
        )
        self.position_embeddings = PositionalEmbedding(
            max_len=max_seq_length, d_model=hidden_size
        )

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

        self.initializer_range = initializer_range
        self.dis_projection = nn.Linear(hidden_size, 1)

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

    def get_extended_attention_mask(self, input_ids):
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

        extended_attention_mask = (
            extended_attention_mask * subsequent_mask
        )  # shape: b*1*max_Sq*max_Sq
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    # model same as SASRec
    def forward(self, input_ids):
        extended_attention_mask = self.get_extended_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        if len(item_encoded_layers) == 0:
            sequence_output = self.intermediate(sequence_emb)
        else:
            sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
