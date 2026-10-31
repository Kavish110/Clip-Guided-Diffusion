import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .imagenet_dic import IMAGENET_DIC


class TINY_IMAGENET_dataset(Dataset):
    """Returns image tensors only (no labels), like the other datasets in this repo.

    Layout: train/<wnid>/images/*.JPEG and val/images/*.JPEG + val/val_annotations.txt.
    `class_num` is an ImageNet-1k class index (see imagenet_dic.py) used to keep a single class.
    """

    def __init__(self, image_root, mode='train', class_num=None, transform=None):
        super().__init__()
        wnid = IMAGENET_DIC[str(class_num)][0] if class_num is not None else None
        if mode == 'train':
            self.image_paths = sorted(glob(os.path.join(image_root, 'train', wnid or '*', 'images', '*.JPEG')))
        else:
            with open(os.path.join(image_root, 'val', 'val_annotations.txt')) as f:
                rows = [line.split('\t')[:2] for line in f]
            self.image_paths = [os.path.join(image_root, 'val', 'images', name)
                                for name, label in rows if wnid is None or label == wnid]
        self.transform = transform

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.image_paths)


def get_tiny_imagenet_dataset(dataset_path, config, class_num=None):
    transform = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = TINY_IMAGENET_dataset(dataset_path, 'train', class_num, transform)
    test_dataset = TINY_IMAGENET_dataset(dataset_path, 'val', class_num, transform)
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise FileNotFoundError(f"No Tiny ImageNet images found under {dataset_path}")

    return train_dataset, test_dataset
