from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_data(dataset, max_length):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["sentence", "idx"]
    )

    tokenized_dataset.set_format("torch")

    return tokenized_dataset