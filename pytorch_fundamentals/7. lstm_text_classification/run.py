from RNN_model import LSTMTextClassifier, Config
from util import build_dataset
import argparse
import torch.optim as optim
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description="LSTM Text Classification")
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')

args = parser.parse_args()




if __name__ == "__main__":
    train_loader, valid_loader = build_dataset(batch_size=args.batch_size)
    print(f"Number of training batches: {len(train_loader)}")

    config = Config(dataset=None, embedding=None)
    model = LSTMTextClassifier(config)

    update_options = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(update_options, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    

    model.to(args.device)
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(args.device), labels.to(args.device)

            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Update learning rate after each epoch
        scheduler.step()

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids, labels = batch
                input_ids, labels = input_ids.to(args.device), labels.to(args.device)

                outputs = model(input_ids)
                loss = criterion(outputs, labels)

                # Compute validation metrics
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        # Calculate validation metrics
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = 100 * valid_correct / valid_total

        # Print epoch results
        print(f"  Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {valid_accuracy:.2f}%")
        print("-" * 60)
