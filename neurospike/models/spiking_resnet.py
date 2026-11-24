import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from .spiking_cnn import LIFNode


class SpikingBasicBlock(nn.Module):
    """Residual Spiking Block with identity/projection shortcut."""
    def __init__(self, in_planes: int, planes: int, stride: int = 1, beta: float = 0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lif1 = LIFNode(beta=beta)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lif2 = LIFNode(beta=beta)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(
        self,
        x: torch.Tensor,
        mem1: Optional[torch.Tensor] = None,
        mem2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.shortcut(x)

        out = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(out, mem1)

        out2 = self.bn2(self.conv2(spk1))
        out2 = out2 + residual
        spk2, mem2 = self.lif2(out2, mem2)

        return spk2, mem1, mem2


class SpikingResNet(nn.Module):
    """
    Spiking Residual Neural Network (SEW-ResNet style identity mappings).
    Achieves high accuracy with ultra-low synaptic operations.
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        block_config: List[int] = [2, 2, 2, 2],
        num_steps: int = 10,
        beta: float = 0.9,
    ):
        super().__init__()
        self.in_planes = 64
        self.num_steps = num_steps
        self.num_classes = num_classes

        # Stem
        self.stem_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_lif = LIFNode(beta=beta)

        # ResNet stages
        self.layer1 = self._make_layer(64, block_config[0], stride=1, beta=beta)
        self.layer2 = self._make_layer(128, block_config[1], stride=2, beta=beta)
        self.layer3 = self._make_layer(256, block_config[2], stride=2, beta=beta)
        self.layer4 = self._make_layer(512, block_config[3], stride=2, beta=beta)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.out_lif = LIFNode(beta=beta)

    def _make_layer(self, planes: int, num_blocks: int, stride: int, beta: float) -> nn.ModuleList:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(SpikingBasicBlock(self.in_planes, planes, stride=s, beta=beta))
            self.in_planes = planes
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected shape: (T, B, C, H, W) or (B, T, C, H, W)
        if x.dim() == 5 and x.size(1) == self.num_steps:
            x = x.transpose(0, 1)

        timesteps, batch_size = x.size(0), x.size(1)

        stem_mem = None
        layer_mems = [[(None, None) for _ in layer] for layer in [self.layer1, self.layer2, self.layer3, self.layer4]]
        out_mem = None
        out_spikes = []

        for t in range(timesteps):
            xt = x[t]
            s = self.stem_bn(self.stem_conv(xt))
            out_spk, stem_mem = self.stem_lif(s, stem_mem)

            # Pass through layers
            for stage_idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for block_idx, block in enumerate(layer):
                    m1, m2 = layer_mems[stage_idx][block_idx]
                    out_spk, m1, m2 = block(out_spk, m1, m2)
                    layer_mems[stage_idx][block_idx] = (m1, m2)

            pooled = self.avgpool(out_spk).flatten(1)
            logits = self.fc(pooled)
            spk_t, out_mem = self.out_lif(logits, out_mem)
            out_spikes.append(spk_t)

        total_spikes = torch.stack(out_spikes, dim=0).sum(dim=0)
        return total_spikes
