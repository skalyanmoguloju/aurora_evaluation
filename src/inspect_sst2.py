from datasets import load_dataset

dataset = load_dataset("glue", "sst2")


print("dataset")
print(dataset)

print("dataset train")
print(dataset["train"][0])

print("dataset validation")
print(dataset["validation"][0])

print("dataset train features")
print(dataset["train"].features)