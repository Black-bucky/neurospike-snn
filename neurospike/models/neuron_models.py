import torch
import torch.nn as nn
from typing import Tuple, Optional


class FastSigmoidSpike(torch.autograd.Function):
    """
    Fast sigmoid surrogate gradient for Heaviside step function.
    Forward: Heaviside step
    Backward: Derivative of x / (1 + |x|)
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, scale: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.scale = scale
        return (input_tensor >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        (input_tensor,) = ctx.saved_tensors
        scale = ctx.scale
        # Surrogate gradient: 1 / (1 + |scale * x|)^2
        grad = grad_output / ((1.0 + torch.abs(scale * input_tensor)) ** 2)
        return grad, None


def fast_sigmoid_spike(x: torch.Tensor, scale: float = 25.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, scale)


class AdaptiveLIF(nn.Module):
    r"""
    Adaptive Leaky Integrate-and-Fire (ALIF) neuron model.
    Includes dynamic threshold adaptation to model spike-frequency adaptation:
        V[t] = \beta * V[t-1] + (1 - \beta) * I[t] - \theta[t-1] * S[t-1]
        \theta[t] = \theta_0 + b[t]
        b[t] = \rho * b[t-1] + \alpha * S[t]
    """
    def __init__(
        self,
        beta: float = 0.9,
        threshold_init: float = 1.0,
        adaptation_rate: float = 0.1,
        adaptation_decay: float = 0.95,
        surrogate_scale: float = 25.0,
    ):
        super().__init__()
        self.beta = beta
        self.threshold_init = threshold_init
        self.adaptation_rate = adaptation_rate
        self.adaptation_decay = adaptation_decay
        self.surrogate_scale = surrogate_scale

    def init_state(self, *size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or torch.device("cpu")
        mem = torch.zeros(size, device=device)
        adapt = torch.zeros(size, device=device)
        return mem, adapt

    def forward(
        self,
        current: torch.Tensor,
        mem: torch.Tensor,
        adapt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        effective_threshold = self.threshold_init + adapt
        mem_decayed = self.beta * mem + current
        
        spike = fast_sigmoid_spike(mem_decayed - effective_threshold, self.surrogate_scale)
        
        # Reset mechanism: soft or hard reset (subtraction)
        new_mem = mem_decayed - spike * effective_threshold
        new_adapt = self.adaptation_decay * adapt + self.adaptation_rate * spike
        
        return spike, new_mem, new_adapt


class IzhikevichNeuron(nn.Module):
    r"""
    Izhikevich simple model of spiking neurons:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        if v >= 30 mV, then v <- c, u <- u + d
    """
    def __init__(self, a: float = 0.02, b: float = 0.2, c: float = -65.0, d: float = 8.0, dt: float = 0.5):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        self.v_thresh = 30.0

    @classmethod
    def from_pattern(cls, pattern: str = "regular_spiking"):
        patterns = {
            "regular_spiking": (0.02, 0.2, -65.0, 8.0),
            "intrinsically_bursting": (0.02, 0.2, -55.0, 4.0),
            "chattering": (0.02, 0.2, -50.0, 2.0),
            "fast_spiking": (0.1, 0.2, -65.0, 2.0),
        }
        if pattern not in patterns:
            raise ValueError(f"Unknown pattern {pattern}. Choose from {list(patterns.keys())}")
        a, b, c, d = patterns[pattern]
        return cls(a=a, b=b, c=c, d=d)

    def init_state(self, *size: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or torch.device("cpu")
        v = torch.full(size, self.c, device=device)
        u = v * self.b
        return v, u

    def forward(self, current: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Two sub-steps for numerical stability (Euler method)
        for _ in range(2):
            dv = (0.04 * (v ** 2) + 5.0 * v + 140.0 - u + current) * (self.dt / 2.0)
            du = (self.a * (self.b * v - u)) * (self.dt / 2.0)
            v = v + dv
            u = u + du

        spike = fast_sigmoid_spike(v - self.v_thresh, scale=1.0)
        
        # Reset
        new_v = torch.where(v >= self.v_thresh, torch.tensor(self.c, device=v.device), v)
        new_u = torch.where(v >= self.v_thresh, u + self.d, u)
        
        return spike, new_v, new_u


class StochasticLIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with probabilistic (stochastic) spike generation.
    P(spike) = sigmoid((V - threshold) / temperature)
    """
    def __init__(self, beta: float = 0.9, threshold: float = 1.0, temperature: float = 0.2):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.temperature = temperature

    def init_state(self, *size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or torch.device("cpu")
        return torch.zeros(size, device=device)

    def forward(self, current: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem_decayed = self.beta * mem + current
        prob = torch.sigmoid((mem_decayed - self.threshold) / self.temperature)
        
        if self.training:
            # Gumbel-Softmax-like or surrogate stochastic sampling
            rand = torch.rand_like(prob)
            spike = fast_sigmoid_spike(prob - rand, scale=10.0)
        else:
            spike = (prob >= 0.5).float()
            
        new_mem = mem_decayed * (1.0 - spike)
        return spike, new_mem
