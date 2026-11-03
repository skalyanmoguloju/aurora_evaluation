import torch
from torch.optim import AdamW
from config import LEARNING_RATE


def create_optimizer(model, learning_rate=LEARNING_RATE):
    return AdamW(
        model.parameters(),
        lr=learning_rate
    )

def train_one_epoch(model, train_loader, optimizer, device):
    # This switches RoBERTa to training mode so it updates model weights during training.
    model.train()

    # We need these counters to calculate average loss and accuracy at the end of the epoch.
    total_loss = 0

    # This processes the training dataset one batch at a time. Each batch contains 8 training reviews.
    for batch in train_loader:
        # This moves each training batch to the same device as RoBERTa. Both must be on MPS.
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    return average_loss