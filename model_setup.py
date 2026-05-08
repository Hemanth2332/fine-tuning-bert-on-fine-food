import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType




MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


def setup_model():

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        target_modules=["query", "value"], 
        bias="none"
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    return model


def compute_metrics(eval_pred):
    
    logits, labels = eval_pred
    predictions = np.argmax(logits,axis=-1)
    
    return {
        "accuracy": accuracy_score(labels,predictions),
        "f1": f1_score(labels,predictions)
    }


def get_training_args():

    return TrainingArguments(
        output_dir="./bert-lora-finefood",
        learning_rate=2e-4,
        
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        
        gradient_accumulation_steps=1,
        
        num_train_epochs=3,
        weight_decay=0.01,

        eval_strategy="epoch",
        save_strategy="epoch",
        # logging_steps=100,
        
        fp16=torch.cuda.is_available(), 
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        
        logging_strategy="steps",
        logging_steps=100,
        report_to="all",
    )