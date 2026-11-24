import torch
import torch.nn as nn
from typing import Optional


class STDPLearner:
    r"""
    Spike-Timing-Dependent Plasticity (STDP) unsupervised learning rule:
        \Delta w = A_+ \exp(-\Delta t / \tau_+)  if \Delta t > 0 (Pre fires before Post -> LTP)
        \Delta w = -A_- \exp(\Delta t / \tau_-)   if \Delta t < 0 (Post fires before Pre -> LTD)
    """
    def __init__(
        self,
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
    ):
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, layer: nn.Linear):
        """
        Applies trace-based online STDP weight update on linear layer.
        pre_spikes: (batch, in_features)
        post_spikes: (batch, out_features)
        """
        with torch.no_grad():
            # LTP: Post spike arrives while pre trace is active
            dw_ltp = torch.matmul(post_spikes.t(), pre_spikes) * self.a_plus
            # LTD: Pre spike arrives while post trace is active
            dw_ltd = torch.matmul(post_spikes.t(), pre_spikes) * self.a_minus

            dw = dw_ltp - dw_ltd
            layer.weight.add_(dw)
            layer.weight.clamp_(self.w_min, self.w_max)


class RewardModulatedSTDP(STDPLearner):
    """
    Reward-Modulated STDP (R-STDP):
    Eligibility trace is stored and converted to permanent synaptic change only
    upon receipt of a global dopamine-like neuromodulatory reward signal:
        \Delta W = Reward * EligibilityTrace
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eligibility_trace = None

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, layer: nn.Linear, reward: float = 1.0):
        with torch.no_grad():
            dw = torch.matmul(post_spikes.t(), pre_spikes) * self.a_plus
            if self.eligibility_trace is None:
                self.eligibility_trace = torch.zeros_like(layer.weight)

            # Decay eligibility trace and accumulate
            self.eligibility_trace = 0.9 * self.eligibility_trace + dw
            
            # Apply modulated update
            layer.weight.add_(reward * self.eligibility_trace)
            layer.weight.clamp_(self.w_min, self.w_max)
