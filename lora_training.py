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
import wandb

MODEL_NAME = "bert-base-uncased"

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
train_dataset.set_format(type="torch",columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


print("Setting up the model....")
model = setup_model()


wandb.init(
    project="finefood-lora",
    name="bert-lora-run",
    config={
        "model": MODEL_NAME,
        "epochs": 3,
        "batch_size": 32,
        "learning_rate": 2e-4,
        "lora_r": 8,
        "lora_alpha": 16
    }
)
wandb.watch(model, log="all")


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
trainer.train(resume_from_checkpoint=True)

model.save_pretrained("./bert-lora-adapter")
tokenizer.save_pretrained("./bert-lora-adapter")


print("Training Completed!")