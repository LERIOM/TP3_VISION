import torch
import tqdm
import time
from visualizations import plot_histogram_times, plot_training_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, early_train_loader, late_train_loader, val_loader, nb_epochs=10, device=device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nb_epochs)

    losses = []
    accuracys = []
    val_losses = []
    times = []

    train_dataset_size = len(early_train_loader.dataset)
    val_dataset_size = len(val_loader.dataset)

    for epoch in range(nb_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        seen_train_samples = 0

        if epoch < nb_epochs * 0.8:
            train_loader = early_train_loader
        else:
            train_loader = late_train_loader

        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            seen_train_samples += batch_size
            progress_bar.set_postfix({"loss": total_loss / seen_train_samples})

        model.eval()
        val_loss = 0.0
        correct = 0
        for images, labels in tqdm.tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save(model.state_dict(), f"models/resnet_like_classifier_epoch_{epoch + 1}.pth")
        
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



    torch.save(model.state_dict(), "models/resnet_like_classifier.pth")

    return losses, accuracys, val_losses, times

if __name__ == "__main__":
    from dataset import build_dataloaders
    from models import FastFoodClassifier, ResNetLikeClassifier
    early_train_loader, late_train_loader, val_loader, test_loader = build_dataloaders("data/fast-food-classification-dataset/Fast Food Classification V2", batch_size=64)
    # model = FastFoodClassifier(num_classes=10)
    model = ResNetLikeClassifier(num_classes=10)
    losses, accuracys, val_losses, times = train(model, early_train_loader, late_train_loader, val_loader, nb_epochs=20)
    plot_training_history(losses, accuracys, val_losses)
    plot_histogram_times(times)