import torch

from ..aurora.model import create_aurora_model


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = create_aurora_model(device)

input_ids = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(8, 128),
    device=device
)

attention_mask = torch.ones(
    8,
    128,
    dtype=torch.long,
    device=device
)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask
)

print("Logits shape:", outputs.logits.shape)