import gc
import torch
import time
import copy
from datetime import datetime
from src.base.model import create_model as create_base_model
from src.lora.model import create_model as create_lora_model
from src.aurora.model import create_model as create_aurora_model
from src.utils.optimizer import create_optimizer
from src.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from src.utils.trainer import train_one_epoch
from src.utils.evaluator import evaluate
from src.peft_type import PeftType
from src.utils.logger import log_results
from src.config import SEED, TRAIN_MODEL, EPOCHS, DEBUG, LEARNING_RATE


def __get_model(peft_type: PeftType, rank: int, alpha: int):
    if peft_type == PeftType.BASE:
        return create_base_model()
    elif peft_type == PeftType.LORA:
        return create_lora_model(rank, alpha)
    elif peft_type == PeftType.TANH_AURORA:
        return create_aurora_model(rank, alpha, False)
    elif peft_type == PeftType.TRAINABLE_TANH_AURORA:
        return create_aurora_model(rank, alpha, True)
    else:
        raise ValueError(f"Unknown PeftType: {peft_type}")

def __get_checkpoint_path(peft_type: PeftType, rank: int):
    if DEBUG: 
        if peft_type == PeftType.BASE:
            return f"models/base_debug_rank_{rank}.pt"
        elif peft_type == PeftType.LORA:
            return f"models/lora_debug_rank_{rank}.pt"
        elif peft_type == PeftType.TANH_AURORA:
            return f"models/tanh_aurora_debug_rank_{rank}.pt"
        elif peft_type == PeftType.TRAINABLE_TANH_AURORA:
            return f"models/trainable_tanh_aurora_debug_rank_{rank}.pt"

    if peft_type == PeftType.BASE:
        return f"models/base_model_rank_{rank}.pt"
    elif peft_type == PeftType.LORA:
        return f"models/lora_model_rank_{rank}.pt"
    elif peft_type == PeftType.TANH_AURORA:
        return f"models/tanh_aurora_model_rank_{rank}.pt"
    elif peft_type == PeftType.TRAINABLE_TANH_AURORA:
        return f"models/trainable_tanh_aurora_model_rank_{rank}.pt"

def __clean_mps_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache() 

def run(peft_type:PeftType, train_loader, validation_loader, rank:int = 0, alpha:int = 0):
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = __get_model(peft_type, rank, alpha)
    

     # Move the model to MPS or CPU
    model.to(device)

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

        best_validation_loss = float("inf")
        best_model_state = None

        for epoch in range(EPOCHS):
            __clean_mps_memory()
            if device.type == "mps":
                torch.mps.synchronize()
                memory_before_training = torch.mps.current_allocated_memory()
            else:
                memory_before_training = 0
            start_time = time.perf_counter()

            training_loss, max_memory_epoch = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device
            )
        
            training_time = time.perf_counter() - start_time
            print("Training time:", training_time, "seconds")

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"training loss: {training_loss}"
            )

            validation_loss, validation_accuracy, validation_f1 = evaluate(
                model,
                validation_loader,
                device
            )

            print("Validation loss:", validation_loss)
            print("Validation accuracy:", validation_accuracy)
            print("Validation F1-score:", validation_f1)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = copy.deepcopy(model.state_dict())

            if device.type == "mps":
                max_memory_epoch_mb = max_memory_epoch / (1024 ** 2)
                additional_peak_mb = max(
                    0.0,
                    (max_memory_epoch - memory_before_training) / (1024 ** 2)
                )
            else:
                max_memory_epoch_mb = 0
                additional_peak_mb = 0

            print("Peak MPS memory during training:", max_memory_epoch_mb, "MB")
            print("Additional peak MPS memory:", additional_peak_mb, "MB")

            # ==========================
            # LOG RESULTS HERE
            # ==========================
            log_results({
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_epochs": EPOCHS,
                "epoch": epoch + 1,
                "peft_type": peft_type.name,
                "rank": rank,
                "alpha": alpha,
                "train": TRAIN_MODEL,
                "debug": DEBUG,
                "learning_rate": LEARNING_RATE,
                "training_time_sec": round(training_time, 2) if TRAIN_MODEL else 0.0,
                "max_memory_epoch_mb": round(max_memory_epoch_mb, 2) if TRAIN_MODEL else 0.0,
                "additional_peak_mb": round(additional_peak_mb, 2) if TRAIN_MODEL else 0.0,
                "validation_loss": round(validation_loss, 4),
                "validation_accuracy": round(validation_accuracy, 4),
                "validation_f1": round(validation_f1, 4),
            })

        model.load_state_dict(best_model_state)

        best_loss, best_accuracy, best_f1 = evaluate(
            model,
            validation_loader,
            device
        )

        print("Best model validation loss:", best_loss)
        print("Best model validation accuracy:", best_accuracy)
        print("Best model validation F1-score:", best_f1)

        save_checkpoint(
            model,
            __get_checkpoint_path(peft_type, rank)
        )

        

    else:
        model = load_checkpoint(
            model,
            __get_checkpoint_path(peft_type, rank),
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

    

    del model, train_loader, validation_loader
    if TRAIN_MODEL:
        del optimizer
    __clean_mps_memory()