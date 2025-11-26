from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch



class TFNSDataset(Dataset):
    def __init__(self,dataset,input_key='input_ids',label_key='label'):
        self.dataset = dataset
        self.input_key = input_key
        self.label_key = label_key

    def __getitem__(self, idx):
        text_ids = torch.tensor(self.dataset[idx][self.input_key])
        text_label = torch.tensor(self.dataset[idx][self.label_key])
        return text_ids, text_label

    def __len__(self):
        return self.dataset.num_rows
    

def build_dataset(batch_size=32):
    
    train_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    valid_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")

    model_name = "ProsusAI/finbert"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],          # column name for this dataset
            truncation=True,
            padding="max_length",
            max_length=128
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    valid_dataset = valid_dataset.map(tokenize_function, batched=True)

    train_data = TFNSDataset(train_dataset)
    valid_data = TFNSDataset(valid_dataset)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

