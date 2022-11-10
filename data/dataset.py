from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import torch
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(
            self,
            root='data/celeba/',
            annotations_file='identity_CelebA.txt',
            img_dir='img_align_celeba',
            transform=None,
            target_transform=None):

        self.root = root
        self.img_labels = pd.read_csv(
            f'{self.root}/{annotations_file}', sep=' ')
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = os.path.join(root, img_dir)

    def __len__(self):
        return len(os.listdir(self.img_dir)) - 1

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def get_dataset(config):

    if not Path('./dataset/celeba/img_align_celeba.zip').is_file():
        raise ValueError('Dataset not downloaded, run `python main.py dl`'
                         'first!')

    dataset = CelebADataset(
        root='./dataset/celeba',
        annotations_file='identity_CelebA.txt',
        img_dir='img_align_celeba',
        transform=transforms.Compose([
            transforms.CenterCrop([128, 128]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(config['VAE_PARAMS']['img_shape'])
        ])
    )

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=64,
                                         shuffle=True,
                                         drop_last=True)

    return loader, dataset
