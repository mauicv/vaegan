import zipfile
from pathlib import Path
import gdown


def download_dataset():
    path = Path('./dataset')
    path.mkdir(exist_ok=True)
    path = Path('./dataset/celeba')
    path.mkdir(exist_ok=True)

    base_url = 'https://storage.googleapis.com/celeba-data-aa'
    dataset_url = f'{base_url}/img_align_celeba.zip'
    annotate_url = f'{base_url}/identity_CelebA.txt'

    target_loc_imgs = Path("./dataset/celeba/img_align_celeba.zip")
    if target_loc_imgs.is_file():
        print('Dataset already downloaded')
    else:
        gdown.download(
            dataset_url,
            target_loc_imgs,
            quiet=False
        )

    target_loc_idents = Path("./dataset/celeba/identity_CelebA.txt")
    if target_loc_idents.is_file():
        print('Identities already downloaded')
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