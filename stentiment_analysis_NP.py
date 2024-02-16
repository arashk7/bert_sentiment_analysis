import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List

# Sample data: A list of texts with their corresponding labels (0: negative, 1: positive)
data = [
    ("This is great!", 1),
    ("Really enjoyed this", 1),
    ("Bad experience", 0),
    ("Not good", 0),
]

# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.25, random_state=42)


class SentimentDataset(Dataset):
    def __init__(self, data: List[tuple], tokenizer: BertTokenizer, max_token_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        text, label = self.data[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_token_len = 128

train_dataset = SentimentDataset(train_data, tokenizer, max_token_len)
val_dataset = SentimentDataset(val_data, tokenizer, max_token_len)



class SentimentModel(pl.LightningModule):
    def __init__(self, n_classes: int = 2, steps_per_epoch: int = None, n_epochs: int = None):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, logits = self(input_ids, attention_mask, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5)



batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = SentimentModel(steps_per_epoch=len(train_loader), n_epochs=4)
trainer = pl.Trainer(max_epochs=4)
trainer.fit(model, train_loader, val_loader)


def predict_sentiment(text, model, tokenizer):
    model.eval()  # Put the model in evaluation mode
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        loss, logits = model(input_ids, attention_mask)
        prediction = torch.argmax(logits, dim=1)
        return prediction.item()

# Example usage
sample_text = "This is great!"
prediction = predict_sentiment(sample_text, model, tokenizer)
print(f"Predicted sentiment: {'Positive' if prediction == 1 else 'Negative'}")
