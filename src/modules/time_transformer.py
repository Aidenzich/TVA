import torch.nn as nn

# from .attention import MultiHeadedAttention
from .time_attention import MultiHeadedTimeAttention
from .utils import SublayerConnection
from .feedforward import PositionwiseFeedForward

# Time-aware Transformer
class TimeTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, d_model, attn_heads, feed_forward_hidden, dropout):
        """
        :param d_model: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*d_model
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedTimeAttention(
            h=attn_heads, d_model=d_model, dropout=dropout
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=d_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, times, mask):

        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, times, mask=mask)
        )

        x = self.output_sublayer(x, self.feed_forward)

        return self.dropout(x)
