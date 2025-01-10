import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from fine_food_custom_dataset import FineFoodDataset
import pandas as pd
from tqdm import tqdm

BATCH_SIZE = 8
LOAD = True
EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 300000


def load_and_split_dataset(file_path):
    df = pd.read_csv(file_path)
    df = df[["Text", "Score"]].dropna()
    df["Score"] = df["Score"] - 1

    df = df.sample(n=SAMPLE_SIZE)

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df["Text"].tolist(), df["Score"].tolist(), test_size=0.2, stratify=df["Score"]
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def train(model, train_loader, optimizer):
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        num_batches = len(train_loader)
        loader = tqdm(train_loader, desc="Training Progress")

        for batch in loader:
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = train_loss / (loader.n + 1)
            loader.set_postfix(loss=avg_loss)

        model_save_path = 'fine_food_paper_model'
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")



def validate(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    print(classification_report(true_labels, predictions, target_names=["1-star", "2-star", "3-star", "4-star", "5-star"]))


if __name__ == "__main__":

    if not LOAD:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
        print("\nLoaded the base model - bert-base-uncased ...")
    else:
        model = BertForSequenceClassification.from_pretrained("./fine_food_paper_model", num_labels=5)
        print("\nLoaded from pretrained ...")


    print(f"\nUsing DEVICE: {DEVICE}")
    model.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_and_split_dataset(r"..\archive\Reviews.csv")
    print(f"\ntrain_texts: {len(train_texts)}, val_texts: {len(val_texts)}, test_texts: {len(test_texts)}")

    print("\nLoading the Dataset ...")
    train_dataset = FineFoodDataset(train_texts, train_labels, tokenizer, head_tail=True)
    val_dataset = FineFoodDataset(val_texts, val_labels, tokenizer, head_tail=True)
    test_dataset = FineFoodDataset(test_texts, test_labels, tokenizer, head_tail=True)


    print("\nLoading the Dataloader ...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    train(model, train_loader, optimizer=optimizer)

    validate(model, test_loader)
    
    

    

