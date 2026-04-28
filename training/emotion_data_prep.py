from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

ROOT_DIR   = Path(__file__).resolve().parent.parent
TRAIN_DIR  = str(ROOT_DIR / "archive" / "train")
TEST_DIR   = str(ROOT_DIR / "archive" / "test")
IMG_SIZE   = (48, 48)
BATCH_SIZE = 64
SEED       = 67


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_base_transforms(img_size=(48, 48)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_augmented_transforms(img_size=(48, 48)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class _SubsetWithTransform(Dataset):
    def __init__(self, subset: Subset, transform):
        self.dataset   = subset.dataset
        self.indices   = subset.indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = self.dataset.loader(path)
        return self.transform(img), label


def load_emotion_datasets(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    img_size=(48, 48),
    batch_size=BATCH_SIZE,
    seed=SEED,
    augment=True,
    val_split=0.15,
    num_workers=0,
):
    set_seed(seed)
    base_transform = get_base_transforms(img_size)
    aug_transform  = get_augmented_transforms(img_size)

    full_train = datasets.ImageFolder(root=train_dir, transform=base_transform)
    class_names = full_train.classes

    total  = len(full_train)
    n_val  = int(val_split * total)
    n_train = total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub = torch.utils.data.random_split(
        full_train, [n_train, n_val], generator=generator
    )

    train_ds = _SubsetWithTransform(train_sub, aug_transform) if augment else train_sub

    pin = torch.cuda.is_available()
    test_ds = datasets.ImageFolder(root=test_dir, transform=base_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_sub,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = load_emotion_datasets()
    print(f"Emotion classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}  min={images.min():.2f}  max={images.max():.2f}")
