import zipfile
from pathlib import Path
import gdown
import shutil


def download_dataset(dl_loc='datasets/'):
    path = Path(dl_loc)
    path.mkdir(exist_ok=True)
    path = path / 'celeba'
    path.mkdir(exist_ok=True)

    base_url = 'https://storage.googleapis.com/celeba-data-aa'
    dataset_url = f'{base_url}/img_align_celeba.zip'
    annotate_url = f'{base_url}/identity_CelebA.txt'

    target_loc_imgs = path / 'img_align_celeba.zip'
    if target_loc_imgs.is_file():
        print('Dataset already downloaded')
    else:
        gdown.download(
            dataset_url,
            str(target_loc_imgs),
            quiet=False
        )

    target_loc_idents = path / 'identity_CelebA.txt'
    if target_loc_idents.is_file():
        print('Identities already downloaded')
    else:
        gdown.download(
            annotate_url,
            str(target_loc_idents),
            quiet=False
        )

def unzip_dataset(path='datasets', target='./'):

    path = Path(path) / 'celeba'
    target_path = Path(target)

    shutil.copyfile(
        path / 'identity_CelebA.txt', 
        Path(target) / 'identity_CelebA.txt'
    )

    target_loc_imgs = path / 'img_align_celeba.zip'
    if not target_path.is_dir():
        with zipfile.ZipFile(target_loc_imgs, 'r') as ziphandler:
            ziphandler.extractall(target_path)
    return str(target_path)