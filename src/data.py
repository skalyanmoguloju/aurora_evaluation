from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from config import (
    MAX_LENGTH,
    BATCH_SIZE,
    DEBUG,
    DEBUG_TRAIN_SIZE,
    DEBUG_VALIDATION_SIZE
)


def create_dataloaders():
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.remove_columns(
        ["sentence", "idx"]
    )

    tokenized_dataset.set_format("torch")

    train_data = tokenized_dataset["train"]
    validation_data = tokenized_dataset["validation"]

    if DEBUG:
        train_data = train_data.select(range(DEBUG_TRAIN_SIZE))
        validation_data = validation_data.select(
            range(DEBUG_VALIDATION_SIZE)
        )

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=BATCH_SIZE
    )

    return train_loader, validation_loader
