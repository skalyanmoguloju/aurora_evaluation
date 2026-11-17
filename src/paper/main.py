from src.peft_type import PeftType
from src.runner import run
from src.paper.config import MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, LORA_RANK, LORA_ALPHA, AURORA_ALPHA, AURORA_RANK
from src.utils.data_loader import create_dataloaders



peft_types = [PeftType.LORA, PeftType.TRAINABLE_TANH_AURORA]


train_loader, validation_loader = create_dataloaders(BATCH_SIZE, MAX_LENGTH)

# print('************************************')
# print(f'Running for Base')
# print('************************************')
# run(PeftType.BASE,
#     train_loader, 
#     validation_loader, 
#     epochs=EPOCHS, 
#     learning_rate=LEARNING_RATE, 
#     seed=SEED, 
#     filepath="results/paper_metrics.csv", 
#     is_paper=True)


# print('************************************')
# print(f'Running for Lora')
# print('************************************')
# run(PeftType.LORA,
#     train_loader, 
#     validation_loader, 
#     epochs=EPOCHS, 
#     learning_rate=LEARNING_RATE, 
#     seed=SEED, 
#     filepath="results/paper_metrics.csv",
#     rank=LORA_RANK,
#     alpha=LORA_ALPHA,
#     is_paper=True)


print('************************************')
print(f'Running for Trainable tanh aurora')
print('************************************')
run(PeftType.TRAINABLE_TANH_AURORA,
    train_loader, 
    validation_loader, 
    epochs=EPOCHS, 
    learning_rate=LEARNING_RATE, 
    seed=SEED, 
    filepath="results/paper_metrics.csv",
    rank=LORA_RANK,
    alpha=LORA_ALPHA,
    is_paper=True)
