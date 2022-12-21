from pathlib import Path
from torch.utils.data import Dataset
from duct.utils.audio import AudioUtil
import pandas as pd
import os
from torchvision.io import read_image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os


def get_dataset(target='celeba', path='./datasets/celeba', batch_size=64, **kwargs):
    return {
        'celeba': get_celeba,
        'fma_small': get_fma_small,
    }[target](path, batch_size, **kwargs)


class CelebADataset(Dataset):
    def __init__(
            self,
            root='./datasets/celeba/',
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


def get_celeba(path='./datasets/celeba', data_shape=(128, 128), batch_size=64):
    path = Path(path) 
    if not (path / 'img_align_celeba').is_dir():
        raise ValueError('Dataset not downloaded or unzipped')

    dataset = CelebADataset(
        root=path,
        annotations_file='identity_CelebA.txt',
        img_dir='img_align_celeba',
        transform=transforms.Compose([
            transforms.CenterCrop([128, 128]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize(data_shape)
        ])
    )

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)

    return loader, dataset


def build_index(path='fma_small'):
    dirs = os.listdir(path)
    index = {}
    count = 0
    for item in dirs:
        if item == 'checksums' or item == 'README.txt' or item == '.DS_Store':
            continue
        files = os.listdir(f'{path}/{item}')
        for filename in files:
            index[count] = f'./{path}/{item}/{filename}'
            count += 1
    return index


class FMASmallDataset(Dataset):
    def __init__(self, path='fma_small', duration=2**13, sr=25000):
        self.index = build_index(path)
        self.duration = duration
        self.sr = sr

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # try:
        aud = AudioUtil.open(self.index[idx], 5*self.duration)
        aud = AudioUtil.rechannel(aud)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.random_portion(aud, self.duration)
        # except RuntimeError as err:
        #     print(err)
        #     n = random.randint(0, len(self))
        #     return self[n]
        return aud[0]


def get_fma_small(
        path='./datasets', 
        batch_size=64, 
        **kwargs):
    path = Path(path) / 'fma_small'
    dataset = FMASmallDataset(
        path=path, 
        **kwargs
    )
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    return loader, dataset
