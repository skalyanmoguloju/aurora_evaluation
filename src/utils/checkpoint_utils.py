import torch

## Save the model locally for faster reruns 
def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

## Load the model from path to device to avoid retraining on every run 
def load_checkpoint(model, path, device):
    model.load_state_dict(
        torch.load(path, map_location=device)
    )
    return model