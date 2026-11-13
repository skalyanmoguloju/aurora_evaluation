import torch.nn as nn
from transformers import AutoModelForSequenceClassification

if __package__:
    from .aurora_layer import AuroRALinear
else:
    from aurora_layer import AuroRALinear


def create_aurora_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    for parameter in model.roberta.parameters():
        parameter.requires_grad = False


    for layer in model.roberta.encoder.layer:
        layer.attention.self.query = AuroRALinear(
            base_layer=layer.attention.self.query,
            rank=8,
            alpha=16,
            dropout=0.1
        )

        layer.attention.self.value = AuroRALinear(
            base_layer=layer.attention.self.value,
            rank=8,
            alpha=16,
            dropout=0.1
        )
    
    model = model.to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_percentage = (
        trainable_parameters / total_parameters
    ) * 100

    print("Trainable parameters:", trainable_parameters)
    print("Total parameters:", total_parameters)
    print("Trainable percentage:", trainable_percentage)

    return model