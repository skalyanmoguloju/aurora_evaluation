from src.peft_type import PeftType
from src.runner import run
from src.config import RANKS
from src.utils.data_loader import create_dataloaders



peft_types = [PeftType.LORA, PeftType.TANH_AURORA, PeftType.TRAINABLE_TANH_AURORA]


train_loader, validation_loader = create_dataloaders()

print('************************************')
print(f'Running for Base')
print('************************************')
run(PeftType.BASE, train_loader, validation_loader)


for rank in RANKS:
    alpha = rank * 2

    print('************************************')
    print(f'Running for Rank: {rank} and Alpha: {alpha}')

    for peft_type in peft_types:
        print('-----------------------------------------')
        print(f'Running for Peft Type: {peft_type.name}')
        run(peft_type, train_loader, validation_loader, rank, alpha)
        print('-----------------------------------------')

    print('************************************')


#base -> aurora
