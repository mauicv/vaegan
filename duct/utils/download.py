import zipfile
from pathlib import Path
import gdown
import shutil
import requests
import tqdm


def download(target='celeba', path='./datasets/'):
    _ = {
        'celeba': download_celeba,
        'fma_small': download_fma_small,
    }[target](path=path)


def unzip(target='celeba', path='./datasets/', target_path=None):
    _ = {
        'celeba': unzip_celeba,
        'fma_small': unzip_fma_small,
    }[target](path=path, target=target_path)


def download_celeba(path='datasets/'):
    path = Path(path)
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


def unzip_celeba(path='./datasets', target='./datasets'):

    path = Path(path) / 'celeba'
    target_path = Path(target)
    target_path.mkdir(exist_ok=True)

    shutil.copyfile(
        path / 'identity_CelebA.txt', 
        target_path / 'identity_CelebA.txt'
    )

    target_loc_imgs = path / 'img_align_celeba.zip'
    with zipfile.ZipFile(target_loc_imgs, 'r') as ziphandler:
        ziphandler.extractall(target_path)

    return str(target_path)


def download_fma_small(path='datasets'):
    path = Path(path)
    path.mkdir(exist_ok=True)
    path = path / 'fma_small'
    path.mkdir(exist_ok=True)

    target_loc_imgs = path / 'fma_small.zip'
    base_url = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
    if target_loc_imgs.is_file():
        print('Dataset already downloaded')
    else:
        download_zip_file(base_url, target_loc_imgs)


def download_zip_file(url, fname):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, 'wb') as f:
            for chunk in tqdm.tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)


def unzip_fma_small(path='./datasets', target='./datasets/fma_small/'):
    path = Path(path) / 'fma_small'
    if target is None:
        target = path / 'fma_small'
    target_path = Path(target)
    target_path.mkdir(exist_ok=True)
    target_loc_imgs = path / 'fma_small.zip'
    with zipfile.ZipFile(target_loc_imgs, 'r') as ziphandler:
        ziphandler.extractall(target_path)
