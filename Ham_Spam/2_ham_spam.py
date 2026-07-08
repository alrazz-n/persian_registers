from datasets import load_from_disk

dataset = load_from_disk("./binary_dataset")

train_dataset = dataset["train"]
dev_dataset = dataset["validation"]
test_dataset = dataset["test"]