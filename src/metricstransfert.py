import os
from pathlib import Path
import sys

PROJECT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))

import matplotlib

if sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score


CURVES_PATH = "learning_curves.png"
REPORT_PATH = "classification_report.txt"
SHOW_CURVES = True


# Calcule la loss, la precision et eventuellement les predictions d'un DataLoader.
def evaluate_model(model, data_loader, criterion, device, return_predictions: bool = False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if return_predictions:
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(preds.cpu().tolist())

    average_loss = running_loss / len(data_loader) if len(data_loader) else 0.0
    accuracy = 100 * correct / total if total else 0.0

    if return_predictions:
        return average_loss, accuracy, all_labels, all_predictions

    return average_loss, accuracy


# Construit le rapport de classification et les metriques globales du test set.
def build_test_metrics(labels, predictions, class_names):
    global_accuracy = accuracy_score(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    return global_accuracy, kappa, report


# Sauvegarde le rapport de classification sur disque.
def save_classification_report(report: str, report_path: str = REPORT_PATH):
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"Rapport sauvegarde dans {report_path}")


# Trace, sauvegarde et peut afficher les courbes d'apprentissage.
def save_learning_curves(
    history,
    curves_path: str = CURVES_PATH,
    show_curves: bool = SHOW_CURVES,
):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning Curves - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(curves_path)
    plt.close()
    print(f"Courbes sauvegardees dans {curves_path}")

    if show_curves:
        if "agg" in matplotlib.get_backend().lower():
            print("Affichage interactif indisponible avec le backend matplotlib actuel.")
            return
        image = plt.imread(curves_path)
        plt.figure(figsize=(10, 4))
        plt.imshow(image)
        plt.axis("off")
        plt.title("Learning Curves")
        plt.show()
