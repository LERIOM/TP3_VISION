from pathlib import Path
import shutil

import kagglehub
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


DATASET_HANDLE = "utkarshsaxenadn/fast-food-classification-dataset"
DATASET_SLUG = DATASET_HANDLE.split("/", maxsplit=1)[1]
SPLIT_ALIASES = {
    "train": ("train",),
    "validation": ("valid", "validation", "val"),
    "test": ("test",),
}


def _find_dataset_root(downloaded_path):
    downloaded_path = Path(downloaded_path)
    candidates = [downloaded_path]
    candidates.extend(child for child in downloaded_path.iterdir() if child.is_dir())

    for candidate in candidates:
        child_names = {child.name.lower() for child in candidate.iterdir() if child.is_dir()}
        if any(alias in child_names for aliases in SPLIT_ALIASES.values() for alias in aliases):
            return candidate

    return downloaded_path


def _find_split_dir(dataset_root, aliases):
    dataset_root = Path(dataset_root)
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.lower() in aliases:
            return child
    return None


def load_dataset(download_dir="data", force_download=False):
    base_dir = Path(download_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    destination_dir = base_dir / DATASET_SLUG

    try:
        downloaded_path = Path(
            kagglehub.dataset_download(
                DATASET_HANDLE,
                output_dir=str(destination_dir),
                force_download=force_download,
            )
        )
    except TypeError:
        cached_path = Path(kagglehub.dataset_download(DATASET_HANDLE, force_download=force_download))
        if force_download and destination_dir.exists():
            shutil.rmtree(destination_dir)
        shutil.copytree(cached_path, destination_dir, dirs_exist_ok=True)
        downloaded_path = destination_dir

    path = _find_dataset_root(downloaded_path)

    print("Path to dataset files:", path)
    return str(path)


def build_dataloaders(path):
    dataset_root = _find_dataset_root(path)
    split_dirs = {
        split_name: _find_split_dir(dataset_root, aliases)
        for split_name, aliases in SPLIT_ALIASES.items()
    }

    missing_splits = [split_name for split_name, split_dir in split_dirs.items() if split_dir is None]
    if missing_splits:
        missing = ", ".join(missing_splits)
        raise FileNotFoundError(
            f"Expected Train/Valid/Test directories under '{dataset_root}', missing: {missing}"
        )

    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(root=str(split_dirs["train"]), transform=train_transforms)
    validation_dataset = ImageFolder(root=str(split_dirs["validation"]), transform=validation_transforms)
    test_dataset = ImageFolder(root=str(split_dirs["test"]), transform=train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    load_dataset("data")