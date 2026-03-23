from contextlib import nullcontext
from pathlib import Path
import torch
import tqdm
import time
from visualizations import plot_histogram_times, plot_training_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_device_backend(device):
    if device.type != "cuda":
        return

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

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

def train(model, train_loader, val_loader, nb_epochs=10, device=device):
    _configure_device_backend(device)

    model.to(device)
    use_channels_last = device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs)
    scaler = _build_grad_scaler(device)
    non_blocking = device.type == "cuda"
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    losses = []
    accuracys = []
    val_losses = []
    times = []

    train_dataset_size = len(train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)

    for epoch in range(nb_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        seen_train_samples = 0

        progress_bar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{nb_epochs}",
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

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            seen_train_samples += batch_size
            if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(train_loader):
                progress_bar.set_postfix({"loss": f"{total_loss / seen_train_samples:.4f}"})

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for images, labels in tqdm.tqdm(val_loader, desc="Validation", mininterval=0.5):
                images = images.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                if use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)

                with _autocast_context(device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save(model.state_dict(), models_dir / f"resnet_like_classifier_epoch_{epoch + 1}.pth")
        
        accuracy = correct / val_dataset_size
        end_time = time.time()
        times.append(end_time - start_time)

        print(
            f"Epoch {epoch + 1}/{nb_epochs}, "
            f"Loss: {total_loss / train_dataset_size}, "
            f"Val Loss: {val_loss / val_dataset_size}, "
            f"Val Accuracy: {accuracy * 100:.4f} %"
            f"Time: {times[-1]:.2f} seconds"
        )
        losses.append(total_loss / train_dataset_size)
        accuracys.append(accuracy)
        val_losses.append(val_loss / val_dataset_size)
        scheduler.step()



    torch.save(model.state_dict(), models_dir / "resnet_like_classifier.pth")

    return losses, accuracys, val_losses, times

if __name__ == "__main__":
    from dataset import build_dataloaders
    from models import FastFoodClassifier, ResNetLikeClassifier
    train_loader, val_loader, test_loader = build_dataloaders(
        "data/fast-food-classification-dataset/Fast Food Classification V2",
        batch_size=64,
    )
    # model = FastFoodClassifier(num_classes=10)
    model = ResNetLikeClassifier(num_classes=10)
    losses, accuracys, val_losses, times = train(model, train_loader, val_loader, nb_epochs=20)
    plot_training_history(losses, accuracys, val_losses)
    plot_histogram_times(times)
