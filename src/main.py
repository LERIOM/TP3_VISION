import argparse
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from dataset import build_dataloaders, load_dataset
from metrics import (
    aggregate_classification_reports,
    aggregate_histories,
    build_classification_metrics,
    ensure_dir,
    evaluate_model,
    format_summary_cell,
    history_to_rows,
    plot_epoch_times,
    plot_learning_curves,
    read_json,
    summarize_scalars,
    write_csv,
    write_json,
    write_text,
)
from models import FastFoodClassifier, ResNetLikeClassifier


DEFAULT_RUNS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_TRANSFER_EPOCHS = 15
DEFAULT_CUSTOM_EPOCHS = 15
DEFAULT_SEED_BASE = 42
DEFAULT_NUM_WORKERS = 8
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR_BY_ARG = {
    "transfer": "transfer_learning",
    "custom": "custom_cnn",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_session_dir(output_root, resume_session):
    resume_path = Path(resume_session)
    if resume_path.is_absolute():
        return resume_path

    candidate_in_output = output_root / resume_path
    if candidate_in_output.exists() or len(resume_path.parts) == 1:
        return candidate_in_output

    return resolve_project_path(resume_session)


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if np is not None:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _configure_device_backend(device):
    if device.type != "cuda":
        return

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _build_grad_scaler(device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def build_resnet50_transforms(weights):
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


def prepare_transfer_dataloaders(dataset_path, batch_size, weights, num_workers):
    train_transform, eval_transform = build_resnet50_transforms(weights)
    return build_dataloaders(
        dataset_path,
        batch_size=batch_size,
        train_transform=train_transform,
        eval_transform=eval_transform,
        num_workers=num_workers,
    )


def build_custom_model(name, num_classes):
    if name == "resnet_like":
        return ResNetLikeClassifier(num_classes=num_classes), "ResNet-like"
    if name == "fast_food":
        return FastFoodClassifier(num_classes=num_classes), "FastFoodClassifier"
    raise ValueError(f"Unsupported custom model: {name}")


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch_index,
    num_epochs,
    model_label,
    scaler=None,
    use_channels_last=False,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    progress_bar = tqdm.tqdm(
        train_loader,
        desc=f"{model_label} - Epoch {epoch_index + 1}/{num_epochs}",
        mininterval=0.5,
    )

    for batch_index, (images, labels) in enumerate(progress_bar, start=1):
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        if use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        predictions = outputs.argmax(dim=1)

        running_loss += loss.item() * batch_size
        total += batch_size
        correct += (predictions == labels).sum().item()

        if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(train_loader):
            progress_bar.set_postfix({
                "loss": f"{running_loss / total:.4f}",
                "acc": f"{100.0 * correct / total:.2f}%",
            })

    return running_loss / total, correct / total


def build_run_summary(
    model_name,
    display_name,
    seed,
    device,
    class_names,
    config,
    history,
    test_results,
    classification_metrics,
):
    total_training_time = float(sum(history["epoch_time_seconds"]))
    mean_epoch_time = total_training_time / len(history["epoch_time_seconds"])

    return {
        "model_name": model_name,
        "display_name": display_name,
        "seed": seed,
        "device": str(device),
        "class_names": class_names,
        "config": config,
        "validation": {
            "final_loss": history["val_loss"][-1],
            "final_accuracy": history["val_accuracy"][-1],
            "best_loss": min(history["val_loss"]),
            "best_accuracy": max(history["val_accuracy"]),
        },
        "test": {
            "loss": test_results["loss"],
            "accuracy": classification_metrics["global_accuracy"],
            "accuracy_percent": classification_metrics["global_accuracy"] * 100.0,
            "kappa": classification_metrics["kappa"],
            "macro_precision": classification_metrics["macro_precision"],
            "macro_recall": classification_metrics["macro_recall"],
            "macro_f1": classification_metrics["macro_f1"],
            "weighted_precision": classification_metrics["weighted_precision"],
            "weighted_recall": classification_metrics["weighted_recall"],
            "weighted_f1": classification_metrics["weighted_f1"],
        },
        "timing": {
            "epoch_time_seconds": history["epoch_time_seconds"],
            "total_training_time_seconds": total_training_time,
            "mean_epoch_time_seconds": mean_epoch_time,
        },
    }


def save_run_outputs(run_dir, model, history, summary, classification_metrics, test_results):
    run_dir = ensure_dir(run_dir)

    torch.save(model.state_dict(), run_dir / "model.pth")
    write_json(run_dir / "history.json", history)
    write_csv(
        run_dir / "epoch_metrics.csv",
        fieldnames=[
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "epoch_time_seconds",
        ],
        rows=history_to_rows(history),
    )
    write_json(run_dir / "metrics.json", summary)
    write_text(run_dir / "classification_report.txt", classification_metrics["report_text"])
    write_json(run_dir / "classification_report.json", classification_metrics["report_dict"])
    write_json(
        run_dir / "predictions.json",
        {
            "labels": test_results["labels"],
            "predictions": test_results["predictions"],
        },
    )

    plot_learning_curves(history, run_dir / "learning_curves.png", title_prefix=summary["display_name"])
    plot_epoch_times(
        history["epoch_time_seconds"],
        run_dir / "epoch_times_histogram.png",
        title=f"{summary['display_name']} - Temps par époque",
    )


def run_custom_experiment(args, dataset_path, device, seed, run_dir):
    set_seed(seed)
    _configure_device_backend(device)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)

    model, display_name = build_custom_model(args.custom_model, num_classes)
    model = model.to(device)
    use_channels_last = device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.custom_lr, weight_decay=args.custom_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.custom_epochs)
    scaler = _build_grad_scaler(device)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_seconds": [],
    }

    print(f"Starting custom CNN run with seed {seed} on {device}")
    for epoch_index in range(args.custom_epochs):
        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_index,
            args.custom_epochs,
            display_name,
            scaler=scaler,
            use_channels_last=use_channels_last,
        )
        val_results = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_results["loss"])
        history["val_accuracy"].append(val_results["accuracy"])
        history["epoch_time_seconds"].append(time.time() - start_time)

        scheduler.step()

        print(
            f"Custom CNN - epoch {epoch_index + 1}/{args.custom_epochs} - "
            f"train loss: {train_loss:.4f} - train acc: {train_accuracy * 100:.2f}% - "
            f"val loss: {val_results['loss']:.4f} - val acc: {val_results['accuracy'] * 100:.2f}%"
        )

    test_results = evaluate_model(model, test_loader, criterion, device, return_predictions=True)
    classification_metrics = build_classification_metrics(
        test_results["labels"],
        test_results["predictions"],
        class_names,
    )

    config = {
        "architecture": args.custom_model,
        "epochs": args.custom_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.custom_lr,
        "weight_decay": args.custom_weight_decay,
    }
    summary = build_run_summary(
        model_name="custom_cnn",
        display_name=display_name,
        seed=seed,
        device=device,
        class_names=class_names,
        config=config,
        history=history,
        test_results=test_results,
        classification_metrics=classification_metrics,
    )
    save_run_outputs(run_dir, model, history, summary, classification_metrics, test_results)

    return {
        "summary": summary,
        "history": history,
        "classification_report": classification_metrics["report_dict"],
    }


def run_transfer_experiment(args, dataset_path, device, seed, run_dir):
    set_seed(seed)
    _configure_device_backend(device)

    weights = ResNet50_Weights.DEFAULT
    train_loader, val_loader, test_loader = prepare_transfer_dataloaders(
        dataset_path,
        batch_size=args.batch_size,
        weights=weights,
        num_workers=args.num_workers,
    )
    class_names = train_loader.dataset.classes
    num_classes = len(class_names)

    model = models.resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    use_channels_last = device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.transfer_lr)
    scaler = _build_grad_scaler(device)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time_seconds": [],
    }

    print(f"Starting transfer learning run with seed {seed} on {device}")
    for epoch_index in range(args.transfer_epochs):
        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_index,
            args.transfer_epochs,
            "ResNet50 Transfer",
            scaler=scaler,
            use_channels_last=use_channels_last,
        )
        val_results = evaluate_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_results["loss"])
        history["val_accuracy"].append(val_results["accuracy"])
        history["epoch_time_seconds"].append(time.time() - start_time)

        print(
            f"Transfer - epoch {epoch_index + 1}/{args.transfer_epochs} - "
            f"train loss: {train_loss:.4f} - train acc: {train_accuracy * 100:.2f}% - "
            f"val loss: {val_results['loss']:.4f} - val acc: {val_results['accuracy'] * 100:.2f}%"
        )

    test_results = evaluate_model(model, test_loader, criterion, device, return_predictions=True)
    classification_metrics = build_classification_metrics(
        test_results["labels"],
        test_results["predictions"],
        class_names,
    )

    config = {
        "architecture": "resnet50",
        "epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.transfer_lr,
        "weights": "ResNet50_Weights.DEFAULT",
        "frozen_backbone": True,
    }
    summary = build_run_summary(
        model_name="transfer_learning",
        display_name="ResNet50 Transfer",
        seed=seed,
        device=device,
        class_names=class_names,
        config=config,
        history=history,
        test_results=test_results,
        classification_metrics=classification_metrics,
    )
    save_run_outputs(run_dir, model, history, summary, classification_metrics, test_results)

    return {
        "summary": summary,
        "history": history,
        "classification_report": classification_metrics["report_dict"],
    }


def build_model_aggregate(run_results):
    summaries = [result["summary"] for result in run_results]
    histories = [result["history"] for result in run_results]
    report_dicts = [result["classification_report"] for result in run_results]

    return {
        "display_name": summaries[0]["display_name"],
        "num_runs": len(run_results),
        "runs": summaries,
        "aggregate_metrics": {
            "test_accuracy": summarize_scalars([summary["test"]["accuracy"] for summary in summaries]),
            "kappa": summarize_scalars([summary["test"]["kappa"] for summary in summaries]),
            "macro_f1": summarize_scalars([summary["test"]["macro_f1"] for summary in summaries]),
            "weighted_f1": summarize_scalars([summary["test"]["weighted_f1"] for summary in summaries]),
            "test_loss": summarize_scalars([summary["test"]["loss"] for summary in summaries]),
            "final_val_accuracy": summarize_scalars([summary["validation"]["final_accuracy"] for summary in summaries]),
            "best_val_accuracy": summarize_scalars([summary["validation"]["best_accuracy"] for summary in summaries]),
            "final_val_loss": summarize_scalars([summary["validation"]["final_loss"] for summary in summaries]),
            "total_training_time_seconds": summarize_scalars(
                [summary["timing"]["total_training_time_seconds"] for summary in summaries]
            ),
            "mean_epoch_time_seconds": summarize_scalars(
                [summary["timing"]["mean_epoch_time_seconds"] for summary in summaries]
            ),
        },
        "aggregate_history": aggregate_histories(histories),
        "aggregate_classification_report": aggregate_classification_reports(report_dicts),
    }


def save_model_aggregate(model_dir, aggregate):
    ensure_dir(model_dir)
    write_json(model_dir / "aggregate_metrics.json", aggregate["aggregate_metrics"])
    write_json(model_dir / "aggregate_history.json", aggregate["aggregate_history"])
    write_json(
        model_dir / "aggregate_classification_report.json",
        aggregate["aggregate_classification_report"],
    )
    write_json(model_dir / "runs_summary.json", aggregate["runs"])

    mean_history = {
        "train_loss": aggregate["aggregate_history"]["train_loss"]["mean"],
        "train_accuracy": aggregate["aggregate_history"]["train_accuracy"]["mean"],
        "val_loss": aggregate["aggregate_history"]["val_loss"]["mean"],
        "val_accuracy": aggregate["aggregate_history"]["val_accuracy"]["mean"],
        "epoch_time_seconds": aggregate["aggregate_history"]["epoch_time_seconds"]["mean"],
    }
    plot_learning_curves(
        mean_history,
        model_dir / "mean_learning_curves.png",
        title_prefix=f"{aggregate['display_name']} Mean",
    )


def load_run_result(run_dir):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "history.json"
    report_path = run_dir / "classification_report.json"

    required_paths = [metrics_path, history_path, report_path]
    if not all(path.exists() for path in required_paths):
        return None

    return {
        "summary": read_json(metrics_path),
        "history": read_json(history_path),
        "classification_report": read_json(report_path),
    }


def load_existing_run_results(model_dir):
    model_dir = Path(model_dir)
    run_results = []

    for run_dir in sorted(model_dir.glob("run_*")):
        run_result = load_run_result(run_dir)
        if run_result is None:
            print(f"Skipping incomplete run directory: {run_dir}")
            continue
        run_results.append(run_result)

    return run_results


def load_existing_model_aggregate(model_dir):
    model_dir = Path(model_dir)
    run_results = load_existing_run_results(model_dir)
    if run_results:
        return build_model_aggregate(run_results)

    aggregate_paths = {
        "aggregate_metrics": model_dir / "aggregate_metrics.json",
        "aggregate_history": model_dir / "aggregate_history.json",
        "aggregate_classification_report": model_dir / "aggregate_classification_report.json",
        "runs": model_dir / "runs_summary.json",
    }
    if not all(path.exists() for path in aggregate_paths.values()):
        return None

    runs = read_json(aggregate_paths["runs"])
    return {
        "display_name": runs[0]["display_name"] if runs else model_dir.name,
        "num_runs": len(runs),
        "runs": runs,
        "aggregate_metrics": read_json(aggregate_paths["aggregate_metrics"]),
        "aggregate_history": read_json(aggregate_paths["aggregate_history"]),
        "aggregate_classification_report": read_json(aggregate_paths["aggregate_classification_report"]),
    }


def build_expected_model_config(model_key, args):
    if model_key == "transfer_learning":
        return {
            "architecture": "resnet50",
            "epochs": args.transfer_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.transfer_lr,
            "weights": "ResNet50_Weights.DEFAULT",
            "frozen_backbone": True,
        }

    if model_key == "custom_cnn":
        return {
            "architecture": args.custom_model,
            "epochs": args.custom_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.custom_lr,
            "weight_decay": args.custom_weight_decay,
        }

    raise ValueError(f"Unsupported model key: {model_key}")


def validate_existing_runs(model_key, existing_run_results, expected_config):
    for run_index, run_result in enumerate(existing_run_results, start=1):
        existing_config = run_result["summary"].get("config", {})
        for key, expected_value in expected_config.items():
            existing_value = existing_config.get(key)
            if existing_value != expected_value:
                raise ValueError(
                    f"Existing run configuration mismatch for {model_key}, run {run_index}, field '{key}': "
                    f"found {existing_value!r}, expected {expected_value!r}. "
                    "Use a new session or remove the existing model output directory before rerunning."
                )


def run_requested_model(model_key, args, dataset_path, device, session_dir):
    model_dir = ensure_dir(Path(session_dir) / model_key)
    existing_run_results = load_existing_run_results(model_dir)

    if existing_run_results:
        validate_existing_runs(model_key, existing_run_results, build_expected_model_config(model_key, args))
        print(f"Found {len(existing_run_results)} completed existing runs for {model_key}")

    if len(existing_run_results) > args.runs:
        raise ValueError(
            f"Session already contains {len(existing_run_results)} completed runs for {model_key}, "
            f"which is greater than the requested {args.runs} runs."
        )

    run_results = list(existing_run_results)
    run_function = run_transfer_experiment if model_key == "transfer_learning" else run_custom_experiment

    for run_index in range(len(run_results), args.runs):
        seed = args.seed_base + run_index
        run_dir = model_dir / f"run_{run_index + 1:02d}"
        print(f"Running {model_key} - run {run_index + 1}/{args.runs} with seed {seed}")
        run_results.append(run_function(args, dataset_path, device, seed, run_dir))

    if not run_results:
        raise ValueError(f"No completed runs available for {model_key}")

    aggregate = build_model_aggregate(run_results)
    save_model_aggregate(model_dir, aggregate)
    return aggregate


def add_model_configuration_lines(lines, aggregate):
    if not aggregate["runs"]:
        return

    model_config = aggregate["runs"][0].get("config", {})
    if not model_config:
        return

    lines.extend([
        "",
        "### Configuration",
        "",
    ])
    for key, value in model_config.items():
        lines.append(f"- {key}: `{value}`")


def build_final_report_markdown(session_name, session_dir, config, aggregated_results):
    lines = [
        "# Final Report",
        "",
        f"- Session: `{session_name}`",
        f"- Created at: `{config['created_at']}`",
        f"- Updated at: `{config['updated_at']}`",
        f"- Dataset path: `{config['dataset_path']}`",
        f"- Output directory: `{session_dir}`",
    ]

    if config.get("resume_session"):
        lines.append(f"- Resume mode: `yes` (`{config['resume_session']}`)")

    if config.get("requested_models"):
        lines.append(f"- Models trained in this invocation: `{', '.join(config['requested_models'])}`")

    lines.extend([
        "- Model-specific hyperparameters can differ; see each model section.",
        "",
        "## Overall comparison",
        "",
        "| Model | Runs | Test accuracy | Kappa | Macro F1 | Weighted F1 | Test loss | Final val accuracy | Training time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for aggregate in aggregated_results.values():
        metrics = aggregate["aggregate_metrics"]
        lines.append(
            "| "
            f"{aggregate['display_name']} | "
            f"{aggregate['num_runs']} | "
            f"{format_summary_cell(metrics['test_accuracy'], scale=100.0, suffix='%')} | "
            f"{format_summary_cell(metrics['kappa'])} | "
            f"{format_summary_cell(metrics['macro_f1'])} | "
            f"{format_summary_cell(metrics['weighted_f1'])} | "
            f"{format_summary_cell(metrics['test_loss'])} | "
            f"{format_summary_cell(metrics['final_val_accuracy'], scale=100.0, suffix='%')} | "
            f"{format_summary_cell(metrics['total_training_time_seconds'], suffix=' s')} |"
        )

    for model_key, aggregate in aggregated_results.items():
        lines.extend([
            "",
            f"## {aggregate['display_name']}",
            "",
            f"- Number of runs: `{aggregate['num_runs']}`",
            f"- Aggregate files: `{Path(model_key)}` inside the session output directory",
        ])

        add_model_configuration_lines(lines, aggregate)

        lines.extend([
            "",
            "### Run details",
            "",
            "| Run | Seed | Test accuracy | Kappa | Test loss | Final val accuracy | Best val accuracy | Total training time |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ])

        for index, summary in enumerate(aggregate["runs"], start=1):
            lines.append(
                "| "
                f"{index} | "
                f"{summary['seed']} | "
                f"{summary['test']['accuracy_percent']:.4f}% | "
                f"{summary['test']['kappa']:.4f} | "
                f"{summary['test']['loss']:.4f} | "
                f"{summary['validation']['final_accuracy'] * 100.0:.4f}% | "
                f"{summary['validation']['best_accuracy'] * 100.0:.4f}% | "
                f"{summary['timing']['total_training_time_seconds']:.2f} s |"
            )

        lines.extend([
            "",
            "### Mean classification report",
            "",
            "| Class | Precision | Recall | F1-score | Support |",
            "| --- | --- | --- | --- | --- |",
        ])

        report = aggregate["aggregate_classification_report"]
        for key, values in report.items():
            if key == "accuracy":
                continue
            if not isinstance(values, dict):
                continue

            precision = format_summary_cell(values.get("precision"))
            recall = format_summary_cell(values.get("recall"))
            f1_score = format_summary_cell(values.get("f1-score"))
            support = values.get("support", "n/a")
            lines.append(f"| {key} | {precision} | {recall} | {f1_score} | {support} |")

    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Run repeated CNN experiments and save metrics.")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Number of runs per model.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for new runs.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--download-dir", default="data", help="Dataset download directory.")
    parser.add_argument("--output-dir", default="output", help="Root directory for experiment outputs.")
    parser.add_argument("--session-name", default=None, help="Optional name for a new output session directory.")
    parser.add_argument(
        "--resume-session",
        default=None,
        help="Existing session name or path to resume and merge with new runs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["transfer", "custom"],
        default=["transfer", "custom"],
        help="Models to train in this invocation.",
    )
    parser.add_argument(
        "--custom-model",
        choices=["resnet_like", "fast_food"],
        default="resnet_like",
        help="Custom CNN architecture to use.",
    )
    parser.add_argument("--transfer-epochs", type=int, default=DEFAULT_TRANSFER_EPOCHS, help="Epochs for transfer.")
    parser.add_argument("--custom-epochs", type=int, default=DEFAULT_CUSTOM_EPOCHS, help="Epochs for custom CNN.")
    parser.add_argument("--transfer-lr", type=float, default=1e-3, help="Learning rate for transfer learning.")
    parser.add_argument("--custom-lr", type=float, default=1e-3, help="Learning rate for the custom CNN.")
    parser.add_argument(
        "--custom-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for the custom CNN optimizer.",
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE, help="Base seed for repeated runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")
    if args.transfer_epochs < 1:
        raise ValueError("--transfer-epochs must be at least 1.")
    if args.custom_epochs < 1:
        raise ValueError("--custom-epochs must be at least 1.")
    if args.resume_session and args.session_name:
        raise ValueError("--resume-session and --session-name cannot be used together.")

    device = get_device()

    download_dir = resolve_project_path(args.download_dir)
    output_root = resolve_project_path(args.output_dir)
    dataset_path = load_dataset(str(download_dir))

    now = datetime.now().isoformat(timespec="seconds")
    existing_config = {}

    if args.resume_session:
        session_dir = resolve_session_dir(output_root, args.resume_session)
        if not session_dir.exists():
            raise FileNotFoundError(f"Resume session not found: {session_dir}")
        session_name = session_dir.name
        config_path = session_dir / "config.json"
        if config_path.exists():
            existing_config = read_json(config_path)
    else:
        session_name = args.session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = ensure_dir(output_root / session_name)

    config = {
        **existing_config,
        "created_at": existing_config.get("created_at", now),
        "updated_at": now,
        "device": str(device),
        "dataset_path": str(dataset_path),
        "resume_session": str(args.resume_session) if args.resume_session else None,
        "requested_models": args.models,
        "runs": args.runs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "transfer_epochs": args.transfer_epochs,
        "custom_epochs": args.custom_epochs,
        "transfer_lr": args.transfer_lr,
        "custom_lr": args.custom_lr,
        "custom_weight_decay": args.custom_weight_decay,
        "seed_base": args.seed_base,
        "custom_model": args.custom_model,
    }
    write_json(session_dir / "config.json", config)

    print(f"Dataset path: {dataset_path}")
    print(f"Device: {device}")
    print(f"Saving outputs to: {session_dir}")

    requested_model_keys = [MODEL_DIR_BY_ARG[model_arg] for model_arg in args.models]
    aggregated_results = {}

    if args.resume_session:
        for model_key in MODEL_DIR_BY_ARG.values():
            if model_key in requested_model_keys:
                continue
            aggregate = load_existing_model_aggregate(session_dir / model_key)
            if aggregate is not None:
                aggregated_results[model_key] = aggregate
                print(f"Reusing existing results for {model_key}")

    for model_arg in args.models:
        model_key = MODEL_DIR_BY_ARG[model_arg]
        aggregated_results[model_key] = run_requested_model(model_key, args, dataset_path, device, session_dir)

    final_summary = {
        "session_name": session_name,
        "session_dir": str(session_dir),
        "config": config,
        "models": aggregated_results,
    }
    write_json(session_dir / "final_report.json", final_summary)
    write_text(
        session_dir / "final_report.md",
        build_final_report_markdown(session_name, session_dir, config, aggregated_results),
    )

    print(f"Final report written to: {session_dir / 'final_report.md'}")


if __name__ == "__main__":
    main()
