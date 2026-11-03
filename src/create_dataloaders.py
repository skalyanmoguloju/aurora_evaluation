from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_batch(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_batch, batched=True)