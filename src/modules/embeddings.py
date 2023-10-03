import torch.nn as nn
from torch import Tensor


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x) -> Tensor:
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(TimeEmbedding, self).__init__()
        self.years_emb = nn.Embedding(2100, embed_size)
        self.months_emb = nn.Embedding(13, embed_size)
        self.days_emb = nn.Embedding(32, embed_size)
        self.seasons_emb = nn.Embedding(5, embed_size)
        self.hour_emb = nn.Embedding(25, embed_size)
        self.dayofweek_emb = nn.Embedding(8, embed_size)

    def forward(self, time_features, time_seqs):
        years, months, days, seasons, hours, _, _, dayofweek = time_seqs
        years = self.years_emb(years)
        months = self.months_emb(months)
        days = self.days_emb(days)
        seasons = self.seasons_emb(seasons)
        hours = self.hour_emb(hours)
        dayofweek = self.dayofweek_emb(dayofweek)

        time_dict = {
            "years": years,
            "months": months,
            "days": days,
            "seasons": seasons,
            "hours": hours,
            "dayofweek": dayofweek,
        }

        time_features_tensor = None
        for t in time_features:
            if time_features_tensor is None:
                time_features_tensor = time_dict[t]
            else:
                time_features_tensor += time_dict[t]

        return time_features_tensor
