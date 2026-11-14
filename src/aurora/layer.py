import torch.nn as nn

if __package__:
    from .adapter import AuroRAAdapter
else:
    from adapter import AuroRAAdapter


class AuroRALinear(nn.Module):
    def __init__(
        self,
        base_layer,
        rank=8,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        self.base_layer = base_layer

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        input_dim = base_layer.in_features
        output_dim = base_layer.out_features

        self.adapter = AuroRAAdapter(
            input_dim=input_dim,
            rank=rank,
            output_dim=output_dim,
            alpha=alpha,
            dropout=dropout
        )


    def forward(self, x):
        base_output = self.base_layer(x)
        adapter_output = self.adapter(x)

        return base_output + adapter_output