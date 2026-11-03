import torch
from sklearn.metrics import f1_score


def evaluate_model(model, validation_loader, device):
    # This switches RoBERTa to evaluation mode so it does not update or randomly change its behavior during validation.
    model.eval()

    #We need these counters to calculate average loss and accuracy at the end.
    total_loss = 0
    correct_predictions = 0
    total_examples = 0

    # These lists will collect predictions and true labels from every validation batch so F1 can be calculated across the full validation set.
    all_predictions = []
    all_labels = []


    # prediction_counts = torch.zeros(2, dtype=torch.long)

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

            # prediction_counts += torch.bincount(
            #     predictions.cpu(),
            #     minlength=2
            # )

            total_loss += outputs.loss.item()
            correct_predictions += (
                predictions == batch["label"]
            ).sum().item()
            total_examples += batch["label"].size(0)

            # This collects all predictions and true labels from every validation batch. 
            # We move them to CPU because sklearn calculates F1 using regular Python values.
            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                batch["label"].cpu().tolist()
            )
            
    
    average_loss = total_loss / len(validation_loader)
    accuracy = correct_predictions / total_examples


    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    # print("Predicted class counts:", prediction_counts.tolist())
    return average_loss, accuracy, f1