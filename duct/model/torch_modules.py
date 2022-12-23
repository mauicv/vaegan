import torch.nn as nn


def get_conv(data_dim=2):
    return {
        1: nn.Conv1d, 
        2: nn.Conv2d, 
        3: nn.Conv3d,
    }.get(
        data_dim, 
        ValueError('data_dim must be 1, 2, or 3')
    )

def get_norm(type='batch', data_dim=2):
    return {
        1: {'batch': nn.BatchNorm1d, 'instance': nn.InstanceNorm1d, None: nn.Identity},
        2: {'batch': nn.BatchNorm2d, 'instance': nn.InstanceNorm2d, None: nn.Identity},
        3: {'batch': nn.BatchNorm3d, 'instance': nn.InstanceNorm3d, None: nn.Identity},
    }.get(
        data_dim,
        ValueError('data_dim must be 1, 2, or 3')
    ).get(
        type,
        ValueError('type must be "batch" or "instance"')
    )


def get_rep_pad(data_dim=2):
    return {
        1: nn.ReplicationPad1d,
        2: nn.ReplicationPad2d,
        3: nn.ReplicationPad3d,
    }.get(
        data_dim,
        ValueError('data_dim must be 1, 2, or 3')
    )

def get_upsample(data_dim=2):
    return {
        1: nn.Upsample,
        2: nn.UpsamplingNearest2d,
    }.get(
        data_dim,
        ValueError('data_dim must be 1 or 2')
    )