import torch

from ..aurora_adapter import AuroRAAdapter

torch.manual_seed(42)

batch_size = 8
sequence_length = 128
hidden_size = 768

adapter = AuroRAAdapter(
    input_dim=hidden_size,
    rank=8,
    output_dim=hidden_size,
    alpha=16,
    dropout=0.1
)

x = torch.randn(
    batch_size,
    sequence_length,
    hidden_size
)

output = adapter(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

loss = output.sum()
loss.backward()

print(
    "Gradient norm for B:",
    adapter.up_proj.weight.grad.norm().item()
)

print(
    "Gradient norm for A:",
    adapter.down_proj.weight.grad.norm().item()
)