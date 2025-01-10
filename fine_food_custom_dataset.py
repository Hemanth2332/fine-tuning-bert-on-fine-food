import torch
from torch.utils.data import Dataset

class FineFoodDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_len=128, head_tail=False):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_tail = head_tail

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        if self.head_tail and len(review.split()) > self.max_len:
            tokens = review.split()
            review = " ".join(tokens[:64] + tokens[-64:])

        encoded = self.tokenizer.encode_plus(
            review,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }