from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DATA_DIR  = str(Path(__file__).resolve().parent.parent / "asl-numbers-alphabet-dataset")
IMG_SIZE  = (64, 64)
BATCH_SIZE = 128
SEED      = 67


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(img_size=(64, 64)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])


def get_augmented_transforms(img_size=(64, 64)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        # No horizontal flip — mirroring a hand changes ASL sign meaning
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.0),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
    ])


class _SubsetWithTransform(Dataset):
    """Wraps a Subset to apply a different transform than the parent dataset."""
    def __init__(self, subset: Subset, transform):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = self.dataset.loader(path)
        return self.transform(img), label


def load_preprocessed_datasets(
    data_dir,
    img_size=(64, 64),
    batch_size=128,
    seed=67,
    augment=False,
    num_workers=0,
    expected_num_classes=None,
):
    set_seed(seed)
    data_dir = Path(data_dir)
    base_transform = get_transforms(img_size)

    full_dataset = datasets.ImageFolder(root=data_dir, transform=base_transform)
    class_names  = full_dataset.classes

    if expected_num_classes is not None and len(class_names) != expected_num_classes:
        print(f"WARNING: Expected {expected_num_classes} classes, found {len(class_names)}.")

    total   = len(full_dataset)
    n_train = int(0.70 * total)
    n_val   = int(0.10 * total)
    n_test  = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    if augment:
        train_ds = _SubsetWithTransform(train_sub, get_augmented_transforms(img_size))
    else:
        train_ds = train_sub

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds,  batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, class_names


def count_samples(dataloader):
    return len(dataloader.dataset)


def show_sample_images(dataloader, class_names, num_images=9):
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(class_names[labels[i].item()])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = load_preprocessed_datasets(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {count_samples(train_loader)}  Val: {count_samples(val_loader)}  Test: {count_samples(test_loader)}")
