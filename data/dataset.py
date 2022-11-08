from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import zipfile
from pathlib import Path
import gdown
import torch
from torchvision import transforms


def download_dataset():
    path = Path('./dataset')
    path.mkdir(exist_ok=True)
    path = Path('./dataset/celeba')
    path.mkdir(exist_ok=True)

    base_url = 'https://storage.googleapis.com/celeba-data-aa/'
    dataset_url = f'{base_url}/img_align_celeba.zip'
    annotate_url = f'{base_url}/identity_CelebA.txt'

    target_loc_imgs = Path("./dataset/celeba/img_align_celeba.zip")
    if target_loc_imgs.is_file():
        print('\n Dataset already downloaded')
    else:
        gdown.download(
            dataset_url,
            target_loc_imgs,
            quiet=False
        )

    target_loc_idents = Path("./dataset/celeba/identity_CelebA.txt")
    if target_loc_idents.is_file():
        print('\n Identities already downloaded')
    else:
        gdown.download(
            annotate_url,
            target_loc_idents,
            quiet=False
        )

    target_loc_unzipped_imgs = Path("./dataset/celeba/img_align_celeba")
    if not target_loc_unzipped_imgs.is_dir():
        with zipfile.ZipFile(target_loc_imgs, 'r') as ziphandler:
            ziphandler.extractall('./dataset/celeba')

    return str(target_loc_imgs)


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
    download_dataset()
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
