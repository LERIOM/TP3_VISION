import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from dataset import build_dataloaders, load_dataset
from metricstransfert import (
    build_test_metrics,
    evaluate_model,
    save_classification_report,
    save_learning_curves,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
EPOCHS = 5
LR = 1e-3
MODEL_PATH = "resnet50_transfer.pth"
LOG_EVERY = 10


# Construit les transformations compatibles avec les poids pré-entraînés de ResNet50.
def build_resnet50_transforms(weights: ResNet50_Weights):
    weights_transforms = weights.transforms()
    normalize = transforms.Normalize(
        mean=weights_transforms.mean,
        std=weights_transforms.std,
    )

    train_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


# Charge le dataset et applique les transformations aux trois splits.
def prepare_dataloaders(download_dir: str, weights: ResNet50_Weights):
    path = load_dataset(download_dir)
    dataloaders = build_dataloaders(path)
    train_transform, eval_transform = build_resnet50_transforms(weights)

    if len(dataloaders) == 3:
        early_train_loader, val_loader, test_loader = dataloaders
        late_train_loader = early_train_loader
    elif len(dataloaders) == 4:
        early_train_loader, late_train_loader, val_loader, test_loader = dataloaders
    else:
        raise ValueError(f"Unexpected number of dataloaders returned: {len(dataloaders)}")

    early_train_loader.dataset.transform = train_transform
    late_train_loader.dataset.transform = train_transform
    val_loader.dataset.transform = eval_transform
    test_loader.dataset.transform = eval_transform

    return early_train_loader, late_train_loader, val_loader, test_loader


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

    for batch_index, (images, labels) in enumerate(train_loader, start=1):
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

        if batch_index == 1 or batch_index % LOG_EVERY == 0 or batch_index == len(train_loader):
            print(f"Batch {batch_index}/{len(train_loader)} - Loss: {loss.item():.4f}")

    average_loss = running_loss / len(train_loader) if len(train_loader) else 0.0
    train_acc = 100 * correct / total if total else 0.0
    return average_loss, train_acc


# Sauvegarde les poids du modèle entraîné sur disque.
def save_model(model: nn.Module, model_path: str = MODEL_PATH):
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé dans {model_path}")


# Orchestre le chargement des données, l'entraînement, l'évaluation et la sauvegarde.
def main():
    weights = ResNet50_Weights.DEFAULT
    early_train_loader, late_train_loader, val_loader, test_loader = prepare_dataloaders("data", weights)
    class_names = early_train_loader.dataset.classes
    num_classes = len(class_names)
    model = build_transfer_model(num_classes, weights)
    criterion, optimizer = build_training_components(model)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print(f"Device: {DEVICE}")
    print(f"Train batches (early): {len(early_train_loader)}")
    print(f"Train batches (late): {len(late_train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    for epoch in range(EPOCHS):
        active_train_loader = early_train_loader if epoch < max(1, EPOCHS // 2) else late_train_loader
        phase = "early" if active_train_loader is early_train_loader else "late"
        print(f"Starting epoch {epoch+1}/{EPOCHS} ({phase})")
        train_loss, train_acc = train_one_epoch(model, active_train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} "
            f"- Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%"
        )

    test_loss, test_acc, test_labels, test_predictions = evaluate_model(
        model,
        test_loader,
        criterion,
        DEVICE,
        return_predictions=True,
    )
    global_accuracy, kappa, report = build_test_metrics(test_labels, test_predictions, class_names)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Global Accuracy: {global_accuracy:.4f}")
    print(f"Cohen Kappa: {kappa:.4f}")
    print("Classification Report:")
    print(report)

    save_learning_curves(history)
    save_classification_report(report)
    save_model(model)


if __name__ == "__main__":
    main()
