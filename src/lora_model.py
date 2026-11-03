import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def create_lora_model(device):
    # Load pretrained RoBERTa with a two-class sentiment classifier
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    # Define how LoRA should be added
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        bias="none",
        task_type="SEQ_CLS"
    )

    # Freeze RoBERTa and add the LoRA adapters
    model = get_peft_model(model, lora_config)

    # Move the model to MPS or CPU
    model.to(device)
    
    # Show trainable and total parameter counts
    model.print_trainable_parameters()

    # for name, parameter in model.named_parameters():
    #     if parameter.requires_grad and "classifier" in name:
    #         print("Trainable:", name, parameter.shape)

    return model