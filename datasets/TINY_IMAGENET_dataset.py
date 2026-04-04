import os
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_tiny_imagenet_dataset(dataset_path, config, class_num=None):
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')

    if not os.path.exists(train_path):
        print(f"Warning: Tiny ImageNet train path {train_path} does not exist.")
        train_dataset = None
    else:
        train_dataset = ImageFolder(train_path, transform=transform)

    if not os.path.exists(val_path):
        print(f"Warning: Tiny ImageNet val path {val_path} does not exist.")
        test_dataset = None
    else:
        test_dataset = ImageFolder(val_path, transform=transform)

    return train_dataset, test_dataset
