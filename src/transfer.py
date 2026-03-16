import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from dataset import build_dataloaders, load_dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
LR = 1e-3
MODEL_PATH = "resnet50_transfer.pth"


# Construit les transformations d'images attendues par ResNet50.
def build_resnet50_transforms(weights: ResNet50_Weights):
    normalize = transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    )

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


# Charge le dataset et applique les transformations aux trois splits.
def prepare_dataloaders(download_dir: str, weights: ResNet50_Weights):
    path = load_dataset(download_dir)
    train_loader, val_loader, test_loader = build_dataloaders(path)
    train_transform, eval_transform = build_resnet50_transforms(weights)

    train_loader.dataset.transform = train_transform
    val_loader.dataset.transform = eval_transform
    test_loader.dataset.transform = eval_transform

    return train_loader, val_loader, test_loader


# Charge ResNet50 pré-entraîné et remplace la couche de classification finale.
def build_transfer_model(num_classes: int, weights: ResNet50_Weights) -> nn.Module:
    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# Crée la fonction de perte et l'optimiseur utilisés pour l'entraînement.
def build_training_components(model: nn.Module):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    return criterion, optimizer


# Exécute une époque d'entraînement complète sur le jeu d'entraînement.
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_acc = 100 * correct / total if total else 0.0
    return running_loss, train_acc


# Calcule la précision du modèle sur un DataLoader donné.
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total if total else 0.0


# Sauvegarde les poids du modèle entraîné sur disque.
def save_model(model: nn.Module, model_path: str = MODEL_PATH):
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé dans {model_path}")


# Orchestre le chargement des données, l'entraînement, l'évaluation et la sauvegarde.
def main():
    weights = ResNet50_Weights.DEFAULT
    train_loader, _, test_loader = prepare_dataloaders("data", weights)
    num_classes = len(train_loader.dataset.classes)
    model = build_transfer_model(num_classes, weights)
    criterion, optimizer = build_training_components(model)

    for epoch in range(EPOCHS):
        running_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss:.4f} - Train Acc: {train_acc:.2f}%")

    test_acc = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")
    save_model(model)


if __name__ == "__main__":
    main()
