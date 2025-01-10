import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from fine_food_custom_dataset import FineFoodDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

model = BertForSequenceClassification.from_pretrained("./fine_food_paper_model", num_labels=5)
model.to(DEVICE)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_and_split_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[["Text", "Score"]].dropna()
    df["Score"] = df["Score"] - 1

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["Text"].tolist(), df["Score"].tolist(), test_size=0.2, stratify=df["Score"]
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

file_path = r"../archive/Reviews.csv"
train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_and_split_dataset(file_path)

sample_texts = test_texts[:5]
sample_labels = [0, 4, 3, 4, 2]  # Updated true labels (different star ratings)

inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(DEVICE)

model.eval()
with torch.no_grad():
    outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()

print("\nSample Test Predictions:")
print("-" * 50)
for i, text in enumerate(sample_texts):
    print(f"Review: {text}")
    print(f"True Label: {sample_labels[i] + 1} stars")
    print(f"Predicted Label: {preds[i] + 1} stars")
    print('-' * 50)
