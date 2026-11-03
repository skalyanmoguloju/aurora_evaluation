import torch
from transformers import AutoModelForSequenceClassification


def create_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    model.to(device)
    return model