import torch.nn as nn

class AuroRAAdapter(nn.Module):
    def __init__(
        self,
        input_dim=768,
        rank=8,
        output_dim=768,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        self.down_proj = nn.Linear(
            input_dim,
            rank,
            bias=False
        )

        self.up_proj = nn.Linear(
            rank,
            output_dim,
            bias=False
        )

        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)
        self.scaling = alpha / rank

        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        update = self.down_proj(x)
        update = self.activation(update)
        update = self.dropout_layer(update)
        update = self.up_proj(update)

        return self.scaling * update