import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_task_metadata():
    return {
        'task_name': 'dl_training_features',
        'task_type': 'classification',
        'num_classes': 10,
        'dataset': 'Fashion-MNIST',
        'features': ['gradient_clipping', 'label_smoothing', 'early_stopping'],
        'description': 'MLP with gradient clipping, label smoothing, early stopping'
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        log_probs = torch.log_softmax(logits, dim=1)
        return -(smooth * log_probs).sum(dim=1).mean()


class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden=256, num_classes=10, dropout=0.3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def make_dataloaders(batch_size=128, val_ratio=0.15, data_root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    full = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
    n = len(full)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def build_model(num_classes=10):
    return MLP(input_dim=784, hidden=256, num_classes=num_classes).to(device)


def train(model, train_loader, val_loader, epochs=50, lr=0.001, max_grad_norm=1.0,
          smoothing=0.1, patience=5):
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')
    epochs_without_improve = 0
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)
        metrics = evaluate(model, val_loader, return_predictions=False)
        val_losses.append(metrics['loss'])
        val_accs.append(metrics['accuracy'])
        epochs_trained = epoch + 1

        if metrics['loss'] < best_val_loss:
            best_val_loss = metrics['loss']
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {avg_train:.4f} Val Loss: {metrics['loss']:.4f} Val Acc: {metrics['accuracy']:.4f}")
        if epochs_without_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accs,
        'epochs_trained': epochs_trained
    }


def evaluate(model, data_loader, return_predictions=True):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    out = {'loss': avg_loss, 'accuracy': acc}
    if return_predictions:
        out['predictions'] = np.array(all_preds)
        out['targets'] = np.array(all_targets)
    return out


def predict(model, X):
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(device)
    with torch.no_grad():
        logits = model(X)
    return logits.cpu().numpy()


def save_artifacts(model, metrics, history, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    to_save = {
        'train_loss': metrics.get('train_loss'),
        'train_accuracy': metrics.get('train_accuracy'),
        'val_loss': metrics.get('val_loss'),
        'val_accuracy': metrics.get('val_accuracy'),
        'epochs_trained': history.get('epochs_trained'),
        'train_losses': history.get('train_losses', []),
        'val_losses': history.get('val_losses', [])
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(to_save, f, indent=2)
    print(f"Artifacts saved to {output_dir}")


def main():
    print("=" * 60)
    print("Training Features: Gradient Clipping, Label Smoothing, Early Stopping")
    print("=" * 60)
    set_seed(42)
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']} | Dataset: {metadata['dataset']}")
    print(f"Device: {get_device()}")

    train_loader, val_loader = make_dataloaders(batch_size=128, val_ratio=0.15)
    model = build_model(num_classes=metadata['num_classes'])
    history = train(
        model, train_loader, val_loader,
        epochs=50, lr=0.001, max_grad_norm=1.0, smoothing=0.1, patience=5
    )

    train_metrics = evaluate(model, train_loader, return_predictions=False)
    val_metrics = evaluate(model, val_loader, return_predictions=True)
    print("\n--- Final metrics ---")
    print(f"Epochs trained: {history['epochs_trained']}")
    print(f"Train Loss: {train_metrics['loss']:.4f}  Train Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val Loss:   {val_metrics['loss']:.4f}  Val Acc:   {val_metrics['accuracy']:.4f}")

    save_artifacts(model, {
        'train_loss': train_metrics['loss'], 'train_accuracy': train_metrics['accuracy'],
        'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy']
    }, history, OUTPUT_DIR)

    try:
        assert val_metrics['accuracy'] > 0.85, f"Val accuracy {val_metrics['accuracy']:.4f} <= 0.85"
        print("\nPASS: Val accuracy > 0.85")
        return 0
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
