from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


TORCH_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "torch"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_CACHE_DIR))


def build_resnet50_transfer_model(
    num_classes: int,
    freeze_backbone: bool = True,
    weights: ResNet50_Weights | None = ResNet50_Weights.DEFAULT,
) -> nn.Module:
    model = resnet50(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_resnet50_transforms(
    weights: ResNet50_Weights = ResNet50_Weights.DEFAULT,
):
    return weights.transforms()


if __name__ == "__main__":
    model = build_resnet50_transfer_model(num_classes=10)
    print(model)
