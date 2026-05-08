from transformers import (
    Trainer,
    DataCollatorWithPadding
)

from model_setup import (
    tokenizer,
    tokenize_function,
    setup_model,
    compute_metrics,
    get_training_args
)

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset



df = pd.read_csv(r"data/balanced_reviews.csv")


train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["labels"],
        random_state=42
    )


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# Tokenizing the text
print("Tokenizing the text.....")
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

# Remove unused columns
train_dataset = train_dataset.remove_columns(
    ["text", "Score"]
)

test_dataset = test_dataset.remove_columns(
    ["text", "Score"]
)

# Torch format
train_dataset.set_format("torch")
test_dataset.set_format("torch")


print("Setting up the model....")
model = setup_model()



print("Setting up the trainer....")
trainer = Trainer(
    model=model,
    args=get_training_args(),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)


print("Training your lora...")
trainer.train()

model.save_pretrained("./bert-lora-adapter")
tokenizer.save_pretrained("./bert-lora-adapter")


print("Training Completed!")