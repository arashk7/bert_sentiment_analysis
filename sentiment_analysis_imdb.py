import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List


from datasets import load_dataset

dataset = load_dataset("imdb")
train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))  # Select a subset for training
test_dataset = dataset['test'].shuffle(seed=42).select(range(200))  # Select a subset for testing




class IMDbDataset(Dataset):
    def __init__(self, data, tokenizer: BertTokenizer, max_token_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        
    def __len__(self):
        return self.data.num_rows
    
    def __getitem__(self, index: int):
        review = self.data[index]['text']
        label = self.data[index]['label']
        encoding = self.tokenizer.encode_plus(
            review,
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




from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_token_len = 128

train_imdb_dataset = IMDbDataset(train_dataset, tokenizer, max_token_len)
test_imdb_dataset = IMDbDataset(test_dataset, tokenizer, max_token_len)




from torch.utils.data import DataLoader

batch_size = 16
train_loader = DataLoader(train_imdb_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_imdb_dataset, batch_size=batch_size)




from pytorch_lightning import Trainer

model = SentimentModel(steps_per_epoch=len(train_loader), n_epochs=4)
trainer = Trainer(max_epochs=4)
trainer.fit(model, train_loader, test_loader)



def predict_sentiment_imdb(text, model, tokenizer):
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
        # Adjusted to match the model's forward method signature, assuming it returns loss and logits
        loss, logits = model(input_ids, attention_mask)
        prediction = torch.argmax(logits, dim=1)
        return prediction.item()

# Assuming 'model' is your trained model and 'tokenizer' is the BertTokenizer instance
sample_text = "This movie was an excellent portrayal of a critical historical event."
prediction = predict_sentiment_imdb(sample_text, model, tokenizer)
print(f"Predicted sentiment: {'Positive' if prediction == 1 else 'Negative'}")



