from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_conv1d_kernel_bn(
    in_channels: int, out_channels: int, kernel_size: int, padding: int
):
    return nn.Sequential(
        nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, groups=in_channels
        ),
        nn.BatchNorm1d(out_channels),
    )


def get_conv1d_k1_bn(
    in_channels: int,
    out_channels: int,
):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 1),
        nn.BatchNorm1d(out_channels),
    )


def trunc(x: Tensor, keep: int):
    """Truncate a Tensor along the last dimension

    Args:
        x (Tensor): tensor to be truncated
        keep (int): number of elements kept
    Returns:
        Tensor: Truncated tensor
    """
    return x[..., :keep]


class SDN(nn.Module):
    def __init__(
        self,
        d_model=8,
        kernel_size=8,
        n_layers=1,
    ):
        super().__init__()
        # in order to fuse this module into next layer, the bias is set False, and no batchnorm follow.
        self.encoder = nn.Conv1d(1, d_model, kernel_size=1, bias=False)
        self.spatial_layers = nn.ModuleList(
            [
                get_conv1d_kernel_bn(
                    d_model, d_model, kernel_size=kernel_size, padding=kernel_size
                )
            ]
        )
        self.feature_layers = nn.ModuleList(
            get_conv1d_k1_bn(d_model, d_model) for _ in range(n_layers)
        )
        for _ in range(n_layers - 1):
            self.spatial_layers.append(
                get_conv1d_kernel_bn(
                    d_model, d_model, kernel_size=kernel_size, padding=kernel_size - 1
                )
            )

        self.decoder = nn.Conv1d(d_model, 1, 1)

    def forward(self, x):
        """
        Input x is shape (B, D, L)
        """
        L = x.size(-1)
        truncL = partial(trunc, keep=L)
        x = self.encoder(x)
        for spatial, features in zip(self.spatial_layers, self.feature_layers):
            x = F.relu(truncL(spatial(x)))
            x = F.relu(features(x) + x)
        return self.decoder(x).squeeze()


class FusedSDN(nn.Module):
    def __init__(
        self,
        d_model=8,
        kernel_size=8,
        n_layers=1,
    ):
        super().__init__()
        # fused module has no encoder, which is fused into the first spatial layer.
        self.encoder = nn.Conv1d(1, d_model, kernel_size=1, padding=1, bias=False)
        self.spatial_layers = nn.ModuleList(
            [nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=kernel_size)]
        )
        self.feature_layers = nn.ModuleList(
            nn.Conv1d(d_model, d_model, 1) for _ in range(n_layers)
        )
        for _ in range(n_layers - 1):
            self.spatial_layers.append(
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    padding=kernel_size - 1,
                    groups=d_model,
                )
            )

        self.decoder = nn.Conv1d(d_model, 1, 1)

    def forward(self, x):
        """
        Input x is shape (B, D, L)
        """
        L = x.size(-1)
        truncL = partial(trunc, keep=L)
        for spatial, features in zip(self.spatial_layers, self.feature_layers):
            x = F.relu(truncL(spatial(x)))
            x = F.relu(features(x) + x)
        return self.decoder(x).squeeze()


if __name__ == "__main__":
    model = SDN()
    x = torch.randn(64, 1, 1024)
    model(x)
