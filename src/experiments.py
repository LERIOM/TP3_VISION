import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from dataset import build_dataloaders
from metrics import (
    build_classification_metrics,
    ensure_dir,
    evaluate_model,
    history_to_rows,
    plot_epoch_times,
    plot_learning_curves,
    write_csv,
    write_json,
    write_text,
)
from models import FastFoodClassifier, ResNetLikeClassifier
from runtime import autocast_context, build_grad_scaler, configure_device_backend, set_seed


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
        with autocast_context(device):
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
    configure_device_backend(device)

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
    scaler = build_grad_scaler(device)

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
    configure_device_backend(device)

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
    scaler = build_grad_scaler(device)

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
