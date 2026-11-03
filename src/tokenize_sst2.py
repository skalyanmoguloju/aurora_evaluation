from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
import torch
from torch.optim import AdamW

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)
model = model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

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

tokenized_dataset = tokenized_dataset.remove_columns(
    ["sentence", "idx"]
)

tokenized_dataset.set_format("torch")

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=8,
    shuffle=True
)

validation_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=8
)

def evaluate_model(model, validation_loader, device):
    # This switches RoBERTa to evaluation mode so it does not update or randomly change its behavior during validation.
    model.eval()

    #We need these counters to calculate average loss and accuracy at the end.
    total_loss = 0
    correct_predictions = 0
    total_examples = 0

    # This tells PyTorch not to calculate gradients during evaluation.
    with torch.no_grad():

        # This processes the validation dataset one batch at a time. Each batch contains 8 validation reviews.
        for batch in validation_loader:
            # This moves each validation batch to the same device as RoBERTa. Both must be on MPS.
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
 
            # This sends each validation batch through RoBERTa and calculates its loss. No weights are updated because we used torch.no_grad().
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]
            )     

            # This selects the class with the higher score for each review, predictions will contain 8 predicted labels for the batch.
            predictions = outputs.logits.argmax(dim=1)  

            total_loss += outputs.loss.item()
            correct_predictions += (
                predictions == batch["label"]
            ).sum().item()
            total_examples += batch["label"].size(0)
    
    average_loss = total_loss / len(validation_loader)
    accuracy = correct_predictions / total_examples

    return average_loss, accuracy

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



validation_loss, validation_accuracy = evaluate_model(
    model,
    validation_loader,
    device
)

print("Validation loss:", validation_loss)
print("Validation accuracy:", validation_accuracy)


training_loss = train_one_epoch(
    model,
    train_loader,
    optimizer,
    device
)

print("Average training loss:", training_loss)


validation_loss, validation_accuracy = evaluate_model(
    model,
    validation_loader,
    device
)

print("Validation loss after epoch:", validation_loss)
print("Validation accuracy after epoch:", validation_accuracy)
