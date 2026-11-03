from data import create_dataloaders
import torch
from lora_model import create_lora_model
from train import create_optimizer, train_one_epoch
from config import SEED, TRAIN_MODEL, EPOCHS
from evaluate import evaluate_model
from checkpoint import save_checkpoint, load_checkpoint 
import time

torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

train_loader, validation_loader = create_dataloaders()

model = create_lora_model(device)

start_time = time.perf_counter()

if device.type == "mps":
    memory_before = torch.mps.current_allocated_memory()
else:
    memory_before = 0

if TRAIN_MODEL:
    optimizer = create_optimizer(model)

    initial_loss, initial_accuracy, initial_f1 = evaluate_model(
        model,
        validation_loader,
        device
    )

    print("Validation loss before training:", initial_loss)
    print("Validation accuracy before training:", initial_accuracy)
    print("Validation F1-score before training:", initial_f1)

    for epoch in range(EPOCHS):
        training_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"training loss: {training_loss}"
        )

    save_checkpoint(
        model,
        "models/lora_debug.pt"
    )
    training_time = time.perf_counter() - start_time

    print("Training time:", training_time, "seconds")
    if device.type == "mps":
        memory_after = torch.mps.current_allocated_memory()
        memory_used_mb = (
            memory_after - memory_before
        ) / (1024 ** 2)
    else:
        memory_used_mb = 0

    print("Additional MPS memory:", memory_used_mb, "MB")

else:
    model = load_checkpoint(
        model,
        "models/lora_debug.pt",
        device
    )

validation_loss, validation_accuracy, validation_f1 = evaluate_model(
    model,
    validation_loader,
    device
)

print("Validation loss:", validation_loss)
print("Validation accuracy:", validation_accuracy)
print("Validation F1-score:", validation_f1)
