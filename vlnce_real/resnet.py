"""ResNet encoders required by the fixed VLN-CE CMA checkpoint.

The depth encoder architecture is derived from Habitat-Lab 0.1.7.  Habitat-Lab
is MIT licensed; its copyright notice is retained in this project's LICENSE.
The RGB encoder reproduces the TorchVision ResNet-50 module layout so the
original checkpoint keys load without requiring TorchVision at runtime.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _conv1x1(
    in_channels: int, out_channels: int, stride: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


def _conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


class _DepthBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        ngroups: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            _conv1x1(inplanes, planes),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            _conv3x3(planes, planes, stride),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            _conv1x1(planes, planes * self.expansion),
            nn.GroupNorm(ngroups, planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.convs(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _DepthResNet50(nn.Module):
    """GroupNorm ResNet-50 used by the VLN depth encoder."""

    def __init__(
        self, in_channels: int = 1, base_planes: int = 32, ngroups: int = 16
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = base_planes
        self.layer1 = self._make_layer(
            planes=base_planes, blocks=3, ngroups=ngroups
        )
        self.layer2 = self._make_layer(
            planes=base_planes * 2,
            blocks=4,
            ngroups=ngroups,
            stride=2,
        )
        self.layer3 = self._make_layer(
            planes=base_planes * 4,
            blocks=6,
            ngroups=ngroups,
            stride=2,
        )
        self.layer4 = self._make_layer(
            planes=base_planes * 8,
            blocks=3,
            ngroups=ngroups,
            stride=2,
        )
        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        ngroups: int,
        stride: int = 1,
    ) -> nn.Sequential:
        output_channels = planes * _DepthBottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != output_channels:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, output_channels, stride),
                nn.GroupNorm(ngroups, output_channels),
            )

        layers = [
            _DepthBottleneck(
                self.inplanes,
                planes,
                ngroups,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.inplanes = output_channels
        for _ in range(1, blocks):
            layers.append(_DepthBottleneck(self.inplanes, planes, ngroups))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class _DepthVisualEncoder(nn.Module):
    """Depth-only equivalent of Habitat-Baselines ResNetEncoder."""

    def __init__(self, input_height: int = 256) -> None:
        super().__init__()
        self._n_input_rgb = 0
        self._n_input_depth = 1
        self.running_mean_and_var = nn.Sequential()
        self.backbone = _DepthResNet50()

        spatial_size = input_height // 2
        final_spatial = int(
            spatial_size * self.backbone.final_spatial_compress
        )
        compression_channels = int(round(2048 / (final_spatial ** 2)))
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, compression_channels),
            nn.ReLU(True),
        )
        self.output_shape = (
            compression_channels,
            final_spatial,
            final_spatial,
        )

    @property
    def is_blind(self) -> bool:
        return False

    def forward(self, observations) -> Tensor:
        depth = observations["depth"].permute(0, 3, 1, 2)
        depth = F.avg_pool2d(depth, 2)
        depth = self.running_mean_and_var(depth)
        depth = self.backbone(depth)
        return self.compression(depth)


class DepthEncoder(nn.Module):
    """CMA depth encoder with learned 4x4 spatial embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = _DepthVisualEncoder(input_height=256)
        self.spatial_embeddings = nn.Embedding(4 * 4, 64)
        self.output_shape = (128 + 64, 4, 4)

    @property
    def is_blind(self) -> bool:
        return False

    def forward(self, observations) -> Tensor:
        depth = self.visual_encoder(observations)
        batch, _, height, width = depth.size()
        spatial = (
            self.spatial_embeddings(
                torch.arange(
                    self.spatial_embeddings.num_embeddings,
                    device=depth.device,
                    dtype=torch.long,
                )
            )
            .view(1, -1, height, width)
            .expand(batch, self.spatial_embeddings.embedding_dim, height, width)
        )
        return torch.cat([depth, spatial], dim=1)


class _RGBBottleneck(nn.Module):
    """TorchVision ResNet bottleneck with stride in the 3x3 convolution."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_rgb_layer(
    inplanes: int, planes: int, blocks: int, stride: int = 1
) -> Tuple[nn.Sequential, int]:
    output_channels = planes * _RGBBottleneck.expansion
    downsample = None
    if stride != 1 or inplanes != output_channels:
        downsample = nn.Sequential(
            _conv1x1(inplanes, output_channels, stride),
            nn.BatchNorm2d(output_channels),
        )

    layers: List[nn.Module] = [
        _RGBBottleneck(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
        )
    ]
    inplanes = output_channels
    for _ in range(1, blocks):
        layers.append(_RGBBottleneck(inplanes, planes))
    return nn.Sequential(*layers), inplanes


class RGBEncoder(nn.Module):
    """ResNet-50 RGB encoder matching the checkpoint's TorchVision keys."""

    def __init__(self) -> None:
        super().__init__()
        inplanes = 64
        layer1, inplanes = _make_rgb_layer(inplanes, 64, 3)
        layer2, inplanes = _make_rgb_layer(inplanes, 128, 4, stride=2)
        layer3, inplanes = _make_rgb_layer(inplanes, 256, 6, stride=2)
        layer4, _ = _make_rgb_layer(inplanes, 512, 3, stride=2)
        self.cnn = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            layer1,
            layer2,
            layer3,
            layer4,
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.spatial_embeddings = nn.Embedding(4 * 4, 64)
        self.output_shape = (2048 + 64, 4, 4)

    @property
    def is_blind(self) -> bool:
        return False

    def forward(self, observations) -> Tensor:
        rgb = observations["rgb"].permute(0, 3, 1, 2)
        rgb = rgb.contiguous().float() / 255.0
        features = self.cnn(rgb)
        batch, _, height, width = features.size()
        spatial = (
            self.spatial_embeddings(
                torch.arange(
                    self.spatial_embeddings.num_embeddings,
                    device=features.device,
                    dtype=torch.long,
                )
            )
            .view(1, -1, height, width)
            .expand(
                batch,
                self.spatial_embeddings.embedding_dim,
                height,
                width,
            )
        )
        return torch.cat([features, spatial], dim=1)
