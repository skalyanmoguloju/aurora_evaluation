import torch
import torch.nn as nn

from ..aurora_layer import AuroRALinear


torch.manual_seed(42)

batch_size = 8
sequence_length = 128
hidden_size = 768

base_layer = nn.Linear(
    hidden_size,
    hidden_size
)

wrapped_layer = AuroRALinear(
    base_layer=base_layer,
    rank=8,
    alpha=16,
    dropout=0.1
)

x = torch.randn(
    batch_size,
    sequence_length,
    hidden_size
)

output = wrapped_layer(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

trainable_parameters = [
    name
    for name, parameter in wrapped_layer.named_parameters()
    if parameter.requires_grad
]

print("Trainable parameters:")
for name in trainable_parameters:
    print(name)