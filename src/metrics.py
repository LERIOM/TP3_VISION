import csv
import json
import os
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

PROJECT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))

import matplotlib

if sys.platform != "darwin" and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
except ModuleNotFoundError:
    accuracy_score = None
    classification_report = None
    cohen_kappa_score = None


def _require_torch():
    if torch is None:
        raise ImportError("PyTorch is required to evaluate models. Install 'torch'.")


def _require_sklearn():
    if accuracy_score is None or classification_report is None or cohen_kappa_score is None:
        raise ImportError("scikit-learn is required to compute classification metrics.")


def _autocast_context(device):
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path, text):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model, data_loader, criterion, device, return_predictions=False):
    _require_torch()

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    non_blocking = getattr(device, "type", str(device)) == "cuda"
    use_channels_last = getattr(device, "type", str(device)) == "cuda"
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with _autocast_context(device):
                outputs = model(images)
                loss = criterion(outputs, labels) if criterion is not None else None
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)
            total += batch_size
            correct += (predictions == labels).sum().item()

            if loss is not None:
                running_loss += loss.item() * batch_size

            if return_predictions:
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

    average_loss = running_loss / total if criterion is not None and total else None
    accuracy = correct / total if total else 0.0

    results = {
        "loss": average_loss,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "num_samples": total,
    }

    if return_predictions:
        results["labels"] = all_labels
        results["predictions"] = all_predictions

    return results


def build_classification_metrics(labels, predictions, class_names):
    _require_sklearn()

    global_accuracy = float(accuracy_score(labels, predictions))
    kappa = float(cohen_kappa_score(labels, predictions))
    report_text = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "global_accuracy": global_accuracy,
        "kappa": kappa,
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "report_text": report_text,
        "report_dict": report_dict,
    }


def plot_learning_curves(history, output_path, title_prefix=""):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    epochs = range(1, len(history["train_loss"]) + 1)
    title_prefix = f"{title_prefix} - " if title_prefix else ""

    train_accuracy = [value * 100.0 for value in history["train_accuracy"]]
    val_accuracy = [value * 100.0 for value in history["val_accuracy"]]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}Learning Curves - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix}Learning Curves - Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_epoch_times(epoch_times, output_path, title="Distribution des temps par époque"):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 5))
    plt.hist(epoch_times, bins=min(10, max(1, len(epoch_times))), edgecolor="black")
    plt.title(title)
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_scalars(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    values = [float(value) for value in values]
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "std": float(std),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_histories(histories):
    if not histories:
        return {}

    keys = histories[0].keys()
    aggregated = {}

    for key in keys:
        series_list = [history[key] for history in histories]
        lengths = {len(series) for series in series_list}
        if len(lengths) != 1:
            raise ValueError(f"Cannot aggregate history '{key}' with different lengths: {sorted(lengths)}")

        aggregated[key] = {
            "mean": [],
            "std": [],
        }

        for index in range(len(series_list[0])):
            values = [float(series[index]) for series in series_list]
            aggregated[key]["mean"].append(float(statistics.fmean(values)))
            aggregated[key]["std"].append(float(statistics.stdev(values)) if len(values) > 1 else 0.0)

    return aggregated


def aggregate_classification_reports(report_dicts):
    if not report_dicts:
        return {}

    aggregated = {}

    for key, first_value in report_dicts[0].items():
        if isinstance(first_value, dict):
            aggregated[key] = {}
            for metric_name, metric_value in first_value.items():
                if metric_name == "support":
                    aggregated[key][metric_name] = metric_value
                    continue

                values = [float(report[key][metric_name]) for report in report_dicts]
                aggregated[key][metric_name] = summarize_scalars(values)
        else:
            values = [float(report[key]) for report in report_dicts]
            aggregated[key] = summarize_scalars(values)

    return aggregated


def history_to_rows(history):
    num_epochs = len(history["train_loss"])
    rows = []

    for epoch_index in range(num_epochs):
        rows.append({
            "epoch": epoch_index + 1,
            "train_loss": history["train_loss"][epoch_index],
            "train_accuracy": history["train_accuracy"][epoch_index],
            "val_loss": history["val_loss"][epoch_index],
            "val_accuracy": history["val_accuracy"][epoch_index],
            "epoch_time_seconds": history["epoch_time_seconds"][epoch_index],
        })

    return rows


def format_summary_cell(summary, scale=1.0, suffix=""):
    if not summary or summary.get("mean") is None:
        return "n/a"

    mean = summary["mean"] * scale
    std = summary["std"] * scale
    return f"{mean:.4f} ± {std:.4f}{suffix}"
