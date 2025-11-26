import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class Config():
    def __init__(self, dataset, embedding):
        self.vocab_size = 0
        self.pad_token_id = 2
        self.embedding_size = 768  # Input size from FinBERT embeddings
        self.hidden_size = 256     # LSTM hidden state size
        self.num_layers = 4
        self.dropout = 0.3
        self.num_classes = 3



class LSTMTextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").bert.embeddings
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout)
        
        
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids) #[B, S, E]

        lstm_out, _ = self.lstm(embeddings)  # lstm_out: [B, S, H*2]

        last_hidden_state = lstm_out[:, -1, :]  

        logits = self.fc(last_hidden_state)

        return logits