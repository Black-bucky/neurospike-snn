import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from .neuron_models import fast_sigmoid_spike


class LIFNode(nn.Module):
    """Clean self-contained Leaky Integrate-and-Fire Node with surrogate gradients."""
    def __init__(self, beta: float = 0.9, threshold: float = 1.0, surrogate_scale: float = 25.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate_scale = surrogate_scale

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.beta * mem + x
        spk = fast_sigmoid_spike(mem - self.threshold, self.surrogate_scale)
        mem = mem * (1.0 - spk)  # Hard reset
        return spk, mem


class SpikingCNN(nn.Module):
    """
    Deep Spiking Convolutional Neural Network designed for neuromorphic event vision
    (e.g., DVS-CIFAR10, N-Caltech101, Prophesee event data).
    """
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        num_steps: int = 10,
        beta: float = 0.9,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = LIFNode(beta=beta)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.lif2 = LIFNode(beta=beta)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv Block 2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lif3 = LIFNode(beta=beta)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.lif4 = LIFNode(beta=beta)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Dense Classifier Head
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.lif_fc1 = LIFNode(beta=beta)
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, num_classes)
        self.lif_out = LIFNode(beta=beta)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, num_steps, in_channels, H, W) or (num_steps, batch, in_channels, H, W)
        Returns:
            out_spikes: Summed output spikes over timesteps of shape (batch, num_classes)
        """
        # Ensure (steps, batch, C, H, W)
        if x.dim() == 5:
            if x.size(1) == self.num_steps:
                x = x.transpose(0, 1)  # (T, B, C, H, W)
        else:
            raise ValueError(f"Expected 5D input tensor, got shape {x.shape}")

        timesteps, batch_size = x.size(0), x.size(1)

        # Initialize membrane states
        mem1 = mem2 = mem3 = mem4 = mem_fc1 = mem_out = None
        spk_out_rec = []
        self.layer_spikes = {
            "conv1": 0.0, "conv2": 0.0, "conv3": 0.0, "conv4": 0.0, "fc1": 0.0, "out": 0.0
        }

        for t in range(timesteps):
            xt = x[t]
            c1 = self.bn1(self.conv1(xt))
            spk1, mem1 = self.lif1(c1, mem1)
            self.layer_spikes["conv1"] += spk1.detach().mean().item()

            c2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(c2, mem2)
            spk2_pool = self.pool1(spk2)
            self.layer_spikes["conv2"] += spk2.detach().mean().item()

            c3 = self.bn3(self.conv3(spk2_pool))
            spk3, mem3 = self.lif3(c3, mem3)
            self.layer_spikes["conv3"] += spk3.detach().mean().item()

            c4 = self.bn4(self.conv4(spk3))
            spk4, mem4 = self.lif4(c4, mem4)
            spk4_pool = self.pool2(spk4)
            self.layer_spikes["conv4"] += spk4.detach().mean().item()

            pooled = self.adaptive_pool(spk4_pool)
            flat = self.flatten(pooled)

            h1 = self.fc1(flat)
            spk_fc1, mem_fc1 = self.lif_fc1(h1, mem_fc1)
            self.layer_spikes["fc1"] += spk_fc1.detach().mean().item()

            drop = self.dropout(spk_fc1)
            h2 = self.fc2(drop)
            spk_out, mem_out = self.lif_out(h2, mem_out)
            self.layer_spikes["out"] += spk_out.detach().mean().item()

            spk_out_rec.append(spk_out)

        # Average layer spike rates
        for k in self.layer_spikes:
            self.layer_spikes[k] /= timesteps

        # Sum spikes over time for voting/classification: (B, num_classes)
        total_spikes = torch.stack(spk_out_rec, dim=0).sum(dim=0)
        return total_spikes

    def compute_spike_stats(self) -> Dict[str, float]:
        return getattr(self, "layer_spikes", {})
