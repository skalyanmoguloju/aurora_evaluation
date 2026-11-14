from src.peft_type import PeftType
from src.runner import run


peft_types = [PeftType.LORA, PeftType.AURORA]
initial_rank = 2
rank_iterator = 4

print('************************************')
print(f'Running for Base')
print('************************************')
run(PeftType.BASE)


for i in range(rank_iterator):
    rank = initial_rank * (2 ** i)
    alpha = rank * 2

    print('************************************')
    print(f'Running for Rank: {rank} and Alpha: {alpha}')

    for peft_type in peft_types:
        print('-----------------------------------------')
        print(f'Running for Peft Type: {peft_type.name}')
        run(peft_type, rank, alpha)
        print('-----------------------------------------')

    print('************************************')


#base -> aurora
