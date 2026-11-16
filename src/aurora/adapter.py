import torch
import torch.nn as nn

class AuroRAAdapter(nn.Module):
    def __init__(
        self,
        input_dim,
        rank,
        output_dim,
        alpha,
        has_trainable_spline=False
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
        self.dropout_layer = nn.Dropout(0.1)
        self.scaling = alpha / rank

        nn.init.zeros_(self.up_proj.weight)

         # Spline setup
        self.has_trainable_spline = has_trainable_spline
        if self.has_trainable_spline:
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


    def spline_basis(self, x):
        basis = (
            (x.unsqueeze(-1) >= self.knots[:-1])
            & (x.unsqueeze(-1) < self.knots[1:])
        ).to(x.dtype)

        for degree in range(1, self.spline_degree + 1):
            left = basis[..., :-1] * (
                (x.unsqueeze(-1) - self.knots[:-(degree + 1)])
                / (
                    self.knots[degree:-1]
                    - self.knots[:-(degree + 1)]
                )
            )

            right = basis[..., 1:] * (
                (self.knots[degree + 1:] - x.unsqueeze(-1))
                / (
                    self.knots[degree + 1:]
                    - self.knots[1:-degree]
                )
            )

            basis = left + right

        return basis

    def forward(self, x):
        downProjection = self.down_proj(x)

        # Fixed AuroRA path
        fixed_update = self.fixed_activation(downProjection)
        fixed_update = self.self_proj(fixed_update)
        fixed_update = self.fixed_activation(fixed_update)
        
        if self.has_trainable_spline:
            basis = self.spline_basis(downProjection)

            spline_update = torch.sum(
                basis * self.spline_weights,
                dim=-1
            )
            
            final_update = fixed_update + spline_update
        else:
            final_update = fixed_update

        final_update = self.dropout_layer(final_update)
        output = self.up_proj(final_update)

        return self.scaling * output