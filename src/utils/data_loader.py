from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.utils.tokenizer import tokenize_data

from src.config import (
    MAX_LENGTH,
    BATCH_SIZE,
    DEBUG,
    DEBUG_TRAIN_SIZE,
    DEBUG_VALIDATION_SIZE
)

## Created the dataloader for the SST2 dataset used for training and validation of the RoBERTa model 
def create_dataloaders(batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    dataset = load_dataset("glue", "sst2")
    
    tokenized_dataset = tokenize_data(dataset, max_length)

    train_data = tokenized_dataset["train"]
    validation_data = tokenized_dataset["validation"]

    ## If DEBUG is true, then train on a small subset of the data for faster debugging 
    if DEBUG:
        train_data = train_data.select(range(DEBUG_TRAIN_SIZE))
        validation_data = validation_data.select(
            range(DEBUG_VALIDATION_SIZE)
        )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size
    )

    return train_loader, validation_loader
