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

def get_instance_norm(data_dim=2):
    return {
        1: nn.InstanceNorm1d,
        2: nn.InstanceNorm2d,
        3: nn.InstanceNorm3d,
    }.get(
        data_dim,
        ValueError('data_dim must be 1, 2, or 3')
    )

def get_batch_norm(data_dim=2):
    return {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d,
    }.get(
        data_dim,
        ValueError('data_dim must be 1, 2, or 3')
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