import torch
from torch.optim import AdamW

def create_optimizer(model, learning_rate):
    return AdamW(
        model.parameters(),
        lr=learning_rate
    )