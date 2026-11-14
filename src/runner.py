import torch
import time
from src.base.model import create_model as create_base_model
from src.lora.model import create_model as create_lora_model
from src.aurora.model import create_model as create_aurora_model
from src.utils.data_loader import create_dataloaders
from src.utils.optimizer import create_optimizer
from src.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from src.utils.trainer import train_one_epoch
from src.utils.evaluator import evaluate
from src.peft_type import PeftType
from src.config import SEED, TRAIN_MODEL, EPOCHS, DEBUG, LEARNING_RATE


def __get_model(peft_type: PeftType, rank: int, alpha: int):
    if peft_type == PeftType.BASE:
        return create_base_model()
    elif peft_type == PeftType.LORA:
        return create_lora_model(rank, alpha)
    elif peft_type == PeftType.AURORA:
        return create_aurora_model(rank, alpha)

def __get_checkpoint_path(peft_type: PeftType, rank: int):
    if DEBUG: 
        if peft_type == PeftType.BASE:
            return f"models/base_debug_rank_{rank}.pt"
        elif peft_type == PeftType.LORA:
            return f"models/lora_debug_rank_{rank}.pt"
        elif peft_type == PeftType.AURORA:
            return f"models/aurora_debug_rank_{rank}.pt"

    if peft_type == PeftType.BASE:
        return f"models/base_model_rank_{rank}.pt"
    elif peft_type == PeftType.LORA:
        return f"models/lora_model_rank_{rank}.pt"
    elif peft_type == PeftType.AURORA:
        return f"models/aurora_model_rank_{rank}.pt"
        

def run(peft_type:PeftType, rank:int = 0, alpha:int = 0):
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if device.type == "mps":
        memory_before = torch.mps.current_allocated_memory()
    else:
        memory_before = 0

    train_loader, validation_loader = create_dataloaders()
    model = __get_model(peft_type, rank, alpha)

     # Move the model to MPS or CPU
    model.to(device)

    start_time = time.perf_counter()

    if TRAIN_MODEL:
        optimizer = create_optimizer(model, LEARNING_RATE)

        initial_loss, initial_accuracy, initial_f1 = evaluate(
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
            __get_checkpoint_path(peft_type, rank)
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
            __get_checkpoint_path(peft_type),
            device
        )

    validation_loss, validation_accuracy, validation_f1 = evaluate(
        model,
        validation_loader,
        device
    )

    print("Validation loss:", validation_loss)
    print("Validation accuracy:", validation_accuracy)
    print("Validation F1-score:", validation_f1)
