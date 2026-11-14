import torch
from transformers import AutoModelForSequenceClassification

from ..aurora.layer import AuroRALinear


model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

# Select the query layer from Transformer layer 0
original_query = model.roberta.encoder.layer[0].attention.self.query

# Replace only that query layer with the AuroRA wrapper
model.roberta.encoder.layer[0].attention.self.query = AuroRALinear(
    base_layer=original_query,
    rank=8,
    alpha=16,
    dropout=0.1
)

input_ids = torch.randint(
    low=0,
    high=model.config.vocab_size,
    size=(8, 128)
)

attention_mask = torch.ones(
    8,
    128,
    dtype=torch.long
)

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask
)

print("Logits shape:", outputs.logits.shape)
print(
    "Layer 0 query type:",
    type(model.roberta.encoder.layer[0].attention.self.query)
)