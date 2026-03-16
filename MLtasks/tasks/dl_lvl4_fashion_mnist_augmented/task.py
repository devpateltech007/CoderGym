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

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FASHION_CLASSES = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


def get_task_metadata():
    return {
        'task_name': 'dl_fashion_mnist_augmented',
        'task_type': 'classification',
        'num_classes': 10,
        'dataset': 'Fashion-MNIST',
        'augmentation': ['RandomCrop', 'RandomHorizontalFlip', 'Normalize'],
        'description': 'CNN on Fashion-MNIST with data augmentation'
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
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def make_dataloaders(batch_size=128, val_ratio=0.15, data_root='./data'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    full_train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_train)
    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    full_val = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform_val)
    val_dataset = torch.utils.data.Subset(full_val, val_ds.indices)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, full_train


def build_model(num_classes=10):
    return SmallCNN(num_classes=num_classes).to(device)


def train(model, train_loader, val_loader, epochs=20, lr=0.001):
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


def save_augmentation_grid(dataset, output_path, num_samples=20, rows=4):
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, generator=torch.Generator().manual_seed(42))
    images, labels = next(iter(loader))
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    cols = (num_samples + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = images[i].squeeze()
            if img.ndim == 3:
                img = img.transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(FASHION_CLASSES[labels[i]], fontsize=8)
        ax.axis('off')
    plt.suptitle('Fashion-MNIST Augmented Samples (RandomCrop, RandomHorizontalFlip, Normalize)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Augmentation grid saved to {output_path}")


def save_artifacts(model, metrics, history, output_dir=None):
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    to_save = {
        'train_loss': metrics.get('train_loss'),
        'train_accuracy': metrics.get('train_accuracy'),
        'val_loss': metrics.get('val_loss'),
        'val_accuracy': metrics.get('val_accuracy'),
        'train_losses': history.get('train_losses', []),
        'val_losses': history.get('val_losses', [])
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(to_save, f, indent=2)
    print(f"Artifacts saved to {output_dir}")


def main():
    print("=" * 60)
    print("Fashion-MNIST with Data Augmentation")
    print("=" * 60)
    set_seed(42)
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']} | Dataset: {metadata['dataset']}")
    print(f"Augmentation: {metadata['augmentation']}")
    print(f"Device: {get_device()}")

    train_loader, val_loader, full_train = make_dataloaders(batch_size=128, val_ratio=0.15)
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    save_augmentation_grid(full_train, os.path.join(OUTPUT_DIR, 'augmented_samples.png'), num_samples=20, rows=4)

    model = build_model(num_classes=metadata['num_classes'])
    history = train(model, train_loader, val_loader, epochs=20, lr=0.001)

    train_metrics = evaluate(model, train_loader, return_predictions=False)
    val_metrics = evaluate(model, val_loader, return_predictions=True)
    print("\n--- Final metrics ---")
    print(f"Train Loss: {train_metrics['loss']:.4f}  Train Acc: {train_metrics['accuracy']:.4f}")
    print(f"Val Loss:   {val_metrics['loss']:.4f}  Val Acc:   {val_metrics['accuracy']:.4f}")

    save_artifacts(model, {
        'train_loss': train_metrics['loss'], 'train_accuracy': train_metrics['accuracy'],
        'val_loss': val_metrics['loss'], 'val_accuracy': val_metrics['accuracy']
    }, history, OUTPUT_DIR)

    try:
        assert val_metrics['accuracy'] > 0.88, f"Val accuracy {val_metrics['accuracy']:.4f} <= 0.88"
        print("\nPASS: Val accuracy > 0.88")
        return 0
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
