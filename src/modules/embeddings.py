import torch.nn as nn


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

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size, max_len, dropout=0.1) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        self.years_emb = nn.Embedding(2100, embed_size)
        self.months_emb = nn.Embedding(13, embed_size)
        self.days_emb = nn.Embedding(32, embed_size)
        self.seasons_emb = nn.Embedding(5, embed_size)
        self.hour_emb = nn.Embedding(25, embed_size)
        self.dayofweek_emb = nn.Embedding(8, embed_size)

    def forward(self, years, months, days, seasons, hours, dayofweek):
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

        for t in time_dict.keys():
            if time_features_tensor is None:
                time_features_tensor = time_dict[t]
            else:
                time_features_tensor = time_features_tensor + time_dict[t]

        time = self.dropout(time_features_tensor)

        return time
