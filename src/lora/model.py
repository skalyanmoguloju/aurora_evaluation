from src.base.model import create_model as create_base_model
from peft import LoraConfig, get_peft_model


def create_model(rank, alpha):
    # Load pretrained RoBERTa with a two-class sentiment classifier
    model = create_base_model()

    # Define how LoRA should be added
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        bias="none",
        task_type="SEQ_CLS"
    )

    # Freeze RoBERTa and add the LoRA adapters
    model = get_peft_model(model, lora_config)

    # Show trainable and total parameter counts
    model.print_trainable_parameters()

    # for name, parameter in model.named_parameters():
    #     if parameter.requires_grad and "classifier" in name:
    #         print("Trainable:", name, parameter.shape)

    return model