import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_task_metadata():
    return {
        'task_name': 'dl_cifar10_cnn',
        'task_type': 'classification',
        'num_classes': 10,
        'input_shape': [3, 32, 32],
        'dataset': 'CIFAR-10',
        'description': 'Small CNN for CIFAR-10 classification'
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def make_dataloaders(batch_size=64, val_ratio=0.2, data_root='./data'):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_dataset, val_dataset = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    full_train_val = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_val)
    val_dataset = torch.utils.data.Subset(full_train_val, val_dataset.indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader


def build_model(num_classes=10):
    model = SmallCNN(num_classes=num_classes).to(device)
    return model


def train(model, train_loader, val_loader, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)
        metrics = evaluate(model, val_loader, return_predictions=False)
        val_losses.append(metrics['loss'])
        val_accs.append(metrics['accuracy'])
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {avg_train:.4f} Val Loss: {metrics['loss']:.4f} Val Acc: {metrics['accuracy']:.4f}")
    return {'train_losses': train_losses, 'val_losses': val_losses, 'val_accuracies': val_accs}


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
    acc = accuracy_score(all_targets, all_preds)
    n = len(all_targets)
    ss_res = sum((np.array(all_targets) - np.array(all_preds)) ** 2)
    ss_tot = sum((np.array(all_targets) - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    metrics = {'loss': avg_loss, 'accuracy': acc, 'mse': float(ss_res / n), 'r2': float(r2)}
    if return_predictions:
        metrics['predictions'] = np.array(all_preds)
        metrics['targets'] = np.array(all_targets)
    return metrics


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
    to_save = {k: v for k, v in metrics.items() if k not in ('predictions', 'targets')}
    if 'val_predictions' in metrics:
        to_save['val_predictions'] = metrics['val_predictions'].tolist()
        to_save['val_targets'] = metrics['val_targets'].tolist()
    to_save['train_losses'] = history.get('train_losses', [])
    to_save['val_losses'] = history.get('val_losses', [])
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(to_save, f, indent=2)
    if 'val_predictions' in metrics and 'val_targets' in metrics:
        cm = confusion_matrix(metrics['val_targets'], metrics['val_predictions'])
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - CIFAR-10')
        plt.colorbar()
        plt.xticks(range(10), CIFAR10_CLASSES, rotation=45, ha='right')
        plt.yticks(range(10), CIFAR10_CLASSES)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Artifacts saved to {output_dir}")


def main():
    print("=" * 60)
    print("CNN on CIFAR-10 - New Dataset Task")
    print("=" * 60)
    set_seed(42)
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']} | Dataset: {metadata['dataset']} | Classes: {metadata['num_classes']}")
    print(f"Device: {get_device()}")

    train_loader, val_loader = make_dataloaders(batch_size=64, val_ratio=0.2)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = build_model(num_classes=metadata['num_classes'])
    history = train(model, train_loader, val_loader, epochs=15, lr=0.001)

    print("\n--- Evaluation ---")
    train_metrics = evaluate(model, train_loader, return_predictions=False)
    val_metrics = evaluate(model, val_loader, return_predictions=True)
    print(f"Train Loss: {train_metrics['loss']:.4f}  Train Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val Loss:   {val_metrics['loss']:.4f}  Val Acc:   {val_metrics['accuracy']:.4f}")
    print(f"Val MSE (classification): {val_metrics.get('mse', 0):.4f}  Val R2: {val_metrics.get('r2', 0):.4f}")

    all_metrics = {
        'train_loss': train_metrics['loss'], 'train_accuracy': train_metrics['accuracy'],
        'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy'],
        'val_predictions': val_metrics['predictions'], 'val_targets': val_metrics['targets']
    }
    save_artifacts(model, all_metrics, history, OUTPUT_DIR)

    print("\n--- Quality assertions ---")
    try:
        assert val_metrics['accuracy'] > 0.55, f"Val accuracy {val_metrics['accuracy']:.4f} <= 0.55"
        print("PASS: Val accuracy > 0.55")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
