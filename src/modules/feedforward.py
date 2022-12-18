import torch.nn as nn

# Feed Forward
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:  # wried, why fusion X 2?
        super(PointWiseFeedForward, self).__init__()
        self.conv_1 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        outputs = self.dropout_2(
            self.conv_2(
                self.relu(self.dropout_1(self.conv_1(inputs.transpose(-1, -2))))
            )
        )
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
