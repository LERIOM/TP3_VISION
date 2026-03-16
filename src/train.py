import torch
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, nb_epochs=10, device=device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    losses = []

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nb_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({"loss": total_loss / ((progress_bar.n + 1) * train_loader.batch_size)})
        
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

        accuracy = correct / len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{nb_epochs}, Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val Accuracy: {accuracy * 100:.4f} %")

    torch.save(model.state_dict(), "models/fast_food_classifier.pth")


if __name__ == "__main__":
    from dataset import build_dataloaders
    from models import FastFoodClassifier
    train_loader, val_loader, test_loader = build_dataloaders("data/fast-food-classification-dataset/Fast Food Classification V2")
    model = FastFoodClassifier(num_classes=10)
    train(model, train_loader, val_loader, nb_epochs=10)