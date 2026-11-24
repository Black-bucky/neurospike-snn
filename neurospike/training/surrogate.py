import torch
import torch.nn as nn
from typing import Tuple, Optional


class FastSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.scale = scale
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        (x,) = ctx.saved_tensors
        scale = ctx.scale
        grad = grad_output / ((1.0 + torch.abs(scale * x)) ** 2)
        return grad, None


class FastSigmoid(nn.Module):
    def __init__(self, scale: float = 25.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FastSigmoidFunction.apply(x, self.scale)


class SuperSpikeFunction(torch.autograd.Function):
    """
    SuperSpike surrogate gradient:
    sigma'(x) = 1 / (1 + beta * |x|)^2
    from Zenke & Ganguli, Neural Computation 2018.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return (x >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        grad = grad_output / ((1.0 + beta * torch.abs(x)) ** 2)
        return grad, None


class SuperSpike(nn.Module):
    def __init__(self, beta: float = 10.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SuperSpikeFunction.apply(x, self.beta)


class StraightThroughEstimator(nn.Module):
    """Straight-Through Estimator (STE) passing gradients unmodified."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spk = (x >= 0.0).float()
        return (spk - x).detach() + x


class SpikeOperator(nn.Module):
    """Generic plug-and-play surrogate gradient operator."""
    def __init__(self, method: str = "fast_sigmoid", scale: float = 25.0):
        super().__init__()
        self.method = method
        if method == "fast_sigmoid":
            self.fn = FastSigmoid(scale=scale)
        elif method == "superspike":
            self.fn = SuperSpike(beta=scale)
        elif method == "ste":
            self.fn = StraightThroughEstimator()
        else:
            raise ValueError(f"Unknown surrogate method {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)
