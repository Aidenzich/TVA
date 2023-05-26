import pytorch_lightning as pl
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.configs import RED_COLOR, END_COLOR, OUTPUT_PATH
from ...modules.embeddings import TokenEmbedding, PositionalEmbedding
from ...modules.feedforward import PositionwiseFeedForward
from ...modules.utils import SCHEDULER
from ...metrics import recalls_and_ndcgs_for_ks, METRICS_KS
from ...models.BERT4Rec.model import BERT
import pickle


class TVAModel(pl.LightningModule):
    def __init__(
        self, num_items: int, model_params: Dict, trainer_config: Dict, data_class
    ) -> None:
        # Inherit from LightningModule to utilize built-in features
        super().__init__()

        # Save hyperparameters into log
        self.save_hyperparameters()

        # Handle trainer config
        self.label_smoothing = trainer_config.get("label_smoothing", 0.0)
        self.ks = trainer_config.get("ks", METRICS_KS)
        self.lr = trainer_config.get("lr", 1e-4)
        self.lr_scheduler = SCHEDULER.get(trainer_config.get("lr_scheduler"), None)
        self.lr_scheduler_args = trainer_config.get("lr_scheduler_args")
        self.lr_scheduler_interval = trainer_config.get("lr_scheduler_interval", "step")
        self.weight_decay = trainer_config.get("weight_decay", 0.0)

        # Handle model config
        user_latent_factor_params = model_params.get("user_latent_factor", None)
        item_latent_factor_params = model_params.get("item_latent_factor", None)
        latent_ff_dim = model_params.get("latent_ff_dim", 0)

        self.max_len = model_params["max_len"]
        self.d_model = model_params["d_model"]
        self.num_mask = model_params.get("num_mask", 1)
        self.use_gate = model_params.get("use_gate", True)
        self.use_softmax_on_item_latent = model_params.get(
            "use_softmax_on_item_latent", None
        )

        userwise_var_dim = 0
        itemwise_var_dim = 0

        use_userwise_var = (
            user_latent_factor_params.get("available", False)
            if user_latent_factor_params is not None
            else False
        )
        use_itemwise_var = (
            item_latent_factor_params.get("available", False)
            if item_latent_factor_params is not None
            else False
        )

        if use_userwise_var:
            print("Using userwise latent factor")
            assert user_latent_factor_params is not None, (
                RED_COLOR + "user_latent_factor_params is None" + END_COLOR
            )
            userwise_var_dim = user_latent_factor_params.get("hidden_dim", 0)
        if use_itemwise_var:
            print("Using itemwise latent factor")
            itemwise_var_dim = item_latent_factor_params.get("hidden_dim", 0)

        # Declare Embedding layers
        self.embedding = TVAEmbedding(
            vocab_size=num_items + 2,
            embed_size=self.d_model,
            max_len=self.max_len,
            dropout=model_params["dropout"],
            user_latent_factor_dim=userwise_var_dim if use_userwise_var else 0,
            item_latent_factor_dim=itemwise_var_dim if use_itemwise_var else 0,
            time_features=model_params.get("time_features", []),
            latent_ff_dim=latent_ff_dim,
            use_gate=self.use_gate,
        )

        # Since we are using BERT as our base model, we can just use BERT class
        self.model = BERT(
            n_layers=model_params["n_layers"],
            d_model=self.d_model,
            heads=model_params["heads"],
            dropout=model_params["dropout"],
            embedding=self.embedding,
        )

        # Declare output(decoder) layer
        self.out = nn.Linear(self.d_model, num_items + 1)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Setting up learning rate scheduler
        if self.lr_scheduler != None:
            lr_schedulers = {
                "scheduler": self.lr_scheduler(optimizer, **self.lr_scheduler_args),
                "interval": self.lr_scheduler_interval,
                "monitor": "train_loss",
            }

            return [optimizer], [lr_schedulers]

        return optimizer

    def forward(
        self,
        item_seq,
        userwise_latent_factor,
        itemwise_latent_factor_seq,
        years,
        months,
        days,
        seasons,
        hours,
        minutes,
        seconds,
        dayofweek,
    ) -> torch.Tensor:
        time_seqs = (
            years,
            months,
            days,
            seasons,
            hours,
            minutes,
            seconds,
            dayofweek,
        )

        x = self.model(
            x=item_seq,
            userwise_latent_factor=userwise_latent_factor,
            itemwise_latent_factor_seq=itemwise_latent_factor_seq,
            time_seqs=time_seqs,
        )

        return self.out(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        main_loss = torch.FloatTensor([0]).to(batch["item_seq_0"].device)
        for idx in range(self.num_mask):
            item_seq = batch[f"item_seq_{idx}"]
            batch_labels = batch[f"labels_{idx}"]

            logits = self.forward(
                item_seq=item_seq,
                userwise_latent_factor=batch[f"userwise_latent_factor_{idx}"],
                itemwise_latent_factor_seq=batch[f"itemwise_latent_factor_seq_{idx}"],
                years=batch[f"years_{idx}"],
                months=batch[f"months_{idx}"],
                days=batch[f"days_{idx}"],
                seasons=batch[f"seasons_{idx}"],
                hours=batch[f"hours_{idx}"],
                minutes=batch[f"minutes_{idx}"],
                seconds=batch[f"seconds_{idx}"],
                dayofweek=batch[f"dayofweek_{idx}"],
            )
            logits = logits.view(-1, logits.size(-1))  # Batch x Seq_len x Item_num
            batch_labels = batch_labels.view(-1)  # Batch x Seq_len

            loss = F.cross_entropy(
                logits,
                batch_labels,
                ignore_index=0,
                label_smoothing=self.label_smoothing,
            )
            main_loss = main_loss + loss

        if self.lr_scheduler != None:
            sch = self.lr_schedulers()
            sch.step()

        self.log("train_loss", main_loss, sync_dist=True)

        return main_loss

    def validation_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )

    def test_step(self, batch, batch_idx) -> None:
        metrics = self.calculate_metrics(batch)

        for metric in metrics.keys():
            if "recall" in metric or "ndcg" in metric:
                self.log(
                    "leave1out_" + metric,
                    torch.FloatTensor([metrics[metric]]),
                    sync_dist=True,
                )

    def calculate_metrics(self, batch) -> Dict[str, float]:
        scores = self.forward(
            item_seq=batch[f"item_seq"],
            userwise_latent_factor=batch[f"userwise_latent_factor"],
            itemwise_latent_factor_seq=batch[f"itemwise_latent_factor_seq"],
            years=batch[f"years"],
            months=batch[f"months"],
            days=batch[f"days"],
            seasons=batch[f"seasons"],
            hours=batch[f"hours"],
            minutes=batch[f"minutes"],
            seconds=batch[f"seconds"],
            dayofweek=batch[f"dayofweek"],
        )

        # scores = self.forward(batch)  # Batch x Seq_len x Item_num
        scores = scores[:, -1, :]  # Batch x Item_num
        scores[:, 0] = -999.999
        scores[
            :, -1
        ] = -999.999  # pad token and mask token should not appear in the logits outpu

        batch_candidates = batch["candidates"]
        batch_labels = batch["labels"]

        scores = scores.gather(1, batch_candidates)  # Batch x Candidates
        metrics = recalls_and_ndcgs_for_ks(scores, batch_labels, self.ks)
        return metrics


class SoftGate(nn.Module):
    def __init__(self, input_size, num_inputs) -> None:
        super(SoftGate, self).__init__()
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.attention = nn.Linear(input_size * num_inputs, num_inputs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs) -> Tensor:
        assert len(inputs) == self.num_inputs
        batch_size, seq_len, _ = inputs[0].size()

        # Concatenate inputs along the feature dimension
        combined_inputs = torch.cat(inputs, dim=-1)
        attention_weights = self.attention(
            combined_inputs.view(batch_size * seq_len, -1)
        )
        attention_weights = self.softmax(attention_weights)

        # Reshape attention weights to match the input shape
        attention_weights = attention_weights.view(
            batch_size, seq_len, self.num_inputs, 1
        )

        # Apply attention weights to each input
        gated_inputs = [
            input * attention_weights[:, :, i] for i, input in enumerate(inputs)
        ]

        # Sum the gated inputs to get the final output
        output = torch.sum(torch.stack(gated_inputs), dim=0)

        # print("\n")
        # print("=" * 50)
        # print(attention_weights.shape)
        # print(attention_weights)
        # mean = attention_weights.mean(dim=(0, 1))
        # print(mean.shape)
        # print(mean)
        # print("=" * 50)

        return output


# TVA Embedding
class TVAEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        max_len,
        user_latent_factor_dim=None,
        item_latent_factor_dim=None,
        latent_ff_dim=128,
        time_features=[],
        dropout=0.1,
        use_softmax_on_item_latent=False,
        use_gate=True,
    ) -> None:
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)

        self.dropout = nn.Dropout(p=dropout)

        # parameters
        self.embed_size = embed_size
        self.max_len = max_len
        self.user_latent_factor_dim = user_latent_factor_dim
        self.item_latent_factor_dim = item_latent_factor_dim
        self.time_features = time_features
        self.latent_ff_dim = latent_ff_dim
        in_dim = embed_size
        features_num = 1

        # boolean parameters
        self.use_softmax_on_item_latent = use_softmax_on_item_latent
        self.use_gate = use_gate

        if self.user_latent_factor_dim != 0:
            self.user_latent_emb = nn.Linear(user_latent_factor_dim * 2, embed_size)
            self.user_latent_emb_ff = PositionwiseFeedForward(
                d_model=embed_size, d_ff=8, dropout=dropout
            )
            in_dim += embed_size
            features_num += 1

        if self.item_latent_factor_dim != 0:
            self.item_latent_emb = nn.Linear(item_latent_factor_dim * 2, embed_size)
            self.item_latent_emb_ff = PositionwiseFeedForward(
                d_model=embed_size, d_ff=latent_ff_dim, dropout=dropout
            )
            in_dim += embed_size
            features_num += 1

        if len(self.time_features) > 0:
            # Time features

            self.years_emb = nn.Embedding(2100, embed_size)
            self.months_emb = nn.Embedding(13, embed_size)
            self.days_emb = nn.Embedding(32, embed_size)
            self.seasons_emb = nn.Embedding(5, embed_size)
            self.hour_emb = nn.Embedding(25, embed_size)
            # self.minute_emb = nn.Embedding(61, embed_size)
            # self.second_emb = nn.Embedding(61, embed_size)
            self.dayofweek_emb = nn.Embedding(8, embed_size)
            features_num += 1
            in_dim += embed_size

        if in_dim != embed_size:
            self.cat_layer = nn.Linear(
                in_dim, embed_size
            )  # if you declare an un-used layer, it still will effect the output value
            self.gate = SoftGate(embed_size, features_num)

    def forward(
        self,
        item_seq,
        userwise_latent_factor,
        itemwise_latent_factor_seq,
        time_seqs=None,
    ):
        x = self.token(item_seq) + self.position(item_seq)
        _cat = [x]

        if self.user_latent_factor_dim != 0:
            assert userwise_latent_factor.shape[1] == self.user_latent_factor_dim * 2, (
                RED_COLOR
                + "user latent factor dim is not correct, please check model config"
                + END_COLOR
            )

            u_mu = F.softmax(
                userwise_latent_factor[:, : self.user_latent_factor_dim], dim=1
            )
            u_sigma = F.softmax(
                userwise_latent_factor[:, self.user_latent_factor_dim :], dim=1
            )

            u_mu = u_mu.unsqueeze(1).repeat(1, self.max_len, 1)
            u_sigma = u_sigma.unsqueeze(1).repeat(1, self.max_len, 1)
            user_latent = u_mu + u_sigma
            user_latent = self.user_latent_emb_ff(user_latent)
            _cat.append(user_latent)

        if self.item_latent_factor_dim != 0:
            assert (
                itemwise_latent_factor_seq.shape[2] == self.item_latent_factor_dim * 2
            ), (
                RED_COLOR
                + f"item latent factor dim ({itemwise_latent_factor_seq.shape[2]})"
                + f" is not match {self.item_latent_factor_dim * 2}, please check model config"
                + END_COLOR
            )

            # Use softmax on item latent factor
            if self.use_softmax_on_item_latent:
                itemwise_latent_factor_seq = F.softmax(
                    itemwise_latent_factor_seq, dim=2
                )

            item_latent = self.item_latent_emb(itemwise_latent_factor_seq)

            # If latent_ff_dim is not 0, use ff layer to reduce the dimension
            if self.latent_ff_dim != 0:
                item_latent = self.item_latent_emb_ff(item_latent)

            _cat.append(item_latent)

        if self.time_features:
            years, months, days, seasons, hours, minutes, seconds, dayofweek = time_seqs

            print(torch.max(days), torch.min(days))

            years = self.years_emb(years)
            months = self.months_emb(months)
            days = self.days_emb(days)

            seasons = self.seasons_emb(seasons)
            hours = self.hour_emb(hours)
            # seconds = self.second_emb(seconds)
            # minutes = self.minute_emb(minutes)

            dayofweek = self.dayofweek_emb(dayofweek)
            time_dict = {
                "years": years,
                "months": months,
                "days": days,
                "seasons": seasons,
                "hours": hours,
                "dayofweek": dayofweek,
                # "seconds": seconds,
                # "minutes": minutes,
            }

            time_features_tensor = None

            for t in self.time_features:
                if time_features_tensor is None:
                    time_features_tensor = time_dict[t]
                else:
                    time_features_tensor = time_features_tensor + time_dict[t]

            _cat.append(time_features_tensor)

        # Len of _cat is 1 means only original bert embedding
        if len(_cat) != 1:
            # If use gate, then use gate to combine all features
            if self.use_gate:
                # print("use gate")
                x = self.gate(_cat)
            # Otherwise, use concat to combine all features
            else:
                x = self.cat_layer(
                    torch.cat(
                        _cat,
                        dim=-1,
                    )
                )

        x = self.dropout(x)

        return x
