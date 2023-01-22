import torch


def get_causal_mask(l):
    mask = torch.tril(torch.ones(l, l))
    masked_indices = mask[None, None, :l, :l] == 0
    return mask, masked_indices


def get_local_image_mask(image_size=(32, 32), patch_size=(6, 6)):
    h, w = image_size
    mask = torch.zeros((h, w, h, w))    
    patch_w, patch_h = patch_size
    for i in range(h):
        for j in range(w):
            top_l_i = i - int(patch_h/2)
            top_l_j = j - int(patch_w/2)
            for ip in range(patch_h):
                for jp in range(patch_w):
                    boundary_cond_i = top_l_i + ip < h and top_l_i + ip >= 0
                    boundary_cond_j = top_l_j + jp < w and top_l_j + jp >= 0
                    boundary_conds = boundary_cond_i and boundary_cond_j 
                    if boundary_conds:
                        if ip < int(patch_h/2) or (ip == int(patch_h/2) and jp <= int(patch_w/2)):
                            mask[i, j, top_l_i + ip, top_l_j + jp] = 1

    flattend_mask = mask.reshape(h * w, h * w)
    indicies = flattend_mask[None, None, :h * w, :h * w] == 0
    return mask, indicies


