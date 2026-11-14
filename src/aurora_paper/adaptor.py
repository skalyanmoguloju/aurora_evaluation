import torch
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

        self.self_proj = nn.Linear(
            rank,
            rank,
            bias=False
        )

        self.fixed_activation = nn.Tanh()

        self.dropout_layer = nn.Dropout(dropout)

        self.num_basis = 8
        self.spline_degree = 3

        knots = torch.linspace(
            -3.0,
            3.0,
            self.num_basis + self.spline_degree + 1
        )

        self.register_buffer(
            "knots",
            knots
        )

        self.spline_weights = nn.Parameter(
            torch.zeros(
                rank,
                self.num_basis
            )
        )


        self.scaling = alpha / rank

        nn.init.zeros_(self.up_proj.weight)

    def spline_basis(self, x):
        

    def forward(self, x):
        update = self.down_proj(x)
        update = self.fixed_activation(update)
        update = self.self_proj(update)
        update = self.fixed_activation(update)
        update = self.dropout_layer(update)
        update = self.up_proj(update)

        return self.scaling * update