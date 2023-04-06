import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention with Time embedding
    """

    def forward(self, query, key, value, time, mask=None, dropout=None):
        query_time = torch.matmul(query, time.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )

        query_key = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )
        scores = query_time + query_key

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


# MultiHeadedAttention
class MultiHeadedTimeAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.time_layers = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = TimeAttention()

        self.dropout = nn.Dropout(p=dropout)
        # self.calculate_mode = "concat"
        self.calculate_mode = "add"

    def forward(self, query, key, value, times: list, mask=None):
        batch_size = query.size(0)

        x_concat = None
        for t in times:
            if self.calculate_mode == "concat":
                q = self.query_linear(query).view(
                    batch_size, self.h, -1, self.d_k
                )  # b s d
                v = self.value_linear(value).view(batch_size, self.h, -1, self.d_k)
                k = self.key_linear(key).view(batch_size, self.h, -1, self.d_k)
                t = self.time_layers(t).view(batch_size, self.h, -1, self.d_k)

                x, attn = self.attention(q, k, v, t, mask=mask, dropout=self.dropout)

                if x_concat is None:
                    x_concat = x
                else:
                    x_concat = torch.cat([x_concat + x], dim=1)

            if self.calculate_mode == "add":
                q = (
                    self.query_linear(query)
                    .view(batch_size, -1, self.h, self.d_k)
                    .transpose(1, 2)
                )

                v = (
                    self.value_linear(value)
                    .view(batch_size, -1, self.h, self.d_k)
                    .transpose(1, 2)
                )

                k = (
                    self.key_linear(key)
                    .view(batch_size, -1, self.h, self.d_k)
                    .transpose(1, 2)
                )

                t = (
                    self.time_layers(t)
                    .view(batch_size, -1, self.h, self.d_k)
                    .transpose(1, 2)
                )

                x, attn = self.attention(q, k, v, t, mask=mask, dropout=self.dropout)

                if x_concat is None:
                    x_concat = x
                else:
                    x_concat = x_concat + x

        x_concat = (
            x_concat.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h * self.d_k)
        )

        return self.output_linear(x_concat)
