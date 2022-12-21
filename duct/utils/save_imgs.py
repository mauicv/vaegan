import matplotlib.pyplot as plt
import numpy as np


def save_img_pairs(self, imgs_1, imgs_2):
    assert len(imgs_1) == len(imgs_2)
    assert imgs_1.shape[0] == 6
    if imgs_1.shape[1] == 3:
        imgs_1 = imgs_1.permute(0, 2, 3, 1)
    if imgs_2.shape[1] == 3:
        imgs_2 = imgs_2.permute(0, 2, 3, 1)
    imgs_1 = imgs_1.detach().cpu().numpy()
    imgs_2 = imgs_2.detach().cpu().numpy()

    _, ax = plt.subplots(ncols=4, nrows=3)
    for ind, (img1, img2) in enumerate(zip(imgs_1, imgs_2)):
        ax[ind // 2, ind % 2].imshow(img1)
        ax[(ind // 2), (ind % 2) + 2].imshow(img2)

    plt.subplots_adjust(wspace=0, hspace=0)
    for a in ax:
        for b in a:
            b.set_xticklabels([])
            b.set_yticklabels([])
            b.set_aspect('equal')
    fname = self.training_artifcat_path / f'{self.iter_count}.png'
    plt.savefig(fname)