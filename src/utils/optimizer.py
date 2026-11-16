import torch
from torch.optim import AdamW

def create_optimizer(model, learning_rate):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(
        trainable_params,
        lr=learning_rate
    )