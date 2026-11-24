import pytest
import torch
from neurospike.models import SpikingCNN, SpikingResNet, SpikingTransformer
from neurospike.models.neuron_models import AdaptiveLIF, IzhikevichNeuron, StochasticLIF
from neurospike.data.encoders import RateCoding, TemporalCoding, DeltaModulation
from neurospike.training.surrogate import FastSigmoid, SuperSpike


def test_adaptive_lif():
    alif = AdaptiveLIF(beta=0.9, threshold_init=1.0)
    mem, adapt = alif.init_state(4, 32)
    current = torch.randn(4, 32) * 2.0
    spk, new_mem, new_adapt = alif(current, mem, adapt)
    assert spk.shape == (4, 32)
    assert new_mem.shape == (4, 32)
    assert new_adapt.shape == (4, 32)


def test_izhikevich_patterns():
    neuron = IzhikevichNeuron.from_pattern("regular_spiking")
    v, u = neuron.init_state(2, 10)
    current = torch.full((2, 10), 10.0)
    spk, new_v, new_u = neuron(current, v, u)
    assert spk.shape == (2, 10)
    assert new_v.shape == (2, 10)


def test_stochastic_lif():
    slif = StochasticLIF(beta=0.85, threshold=1.0)
    mem = slif.init_state(4, 16)
    current = torch.ones(4, 16) * 1.5
    spk, new_mem = slif(current, mem)
    assert spk.shape == (4, 16)


def test_spiking_cnn_forward():
    model = SpikingCNN(in_channels=2, num_classes=10, num_steps=4)
    x = torch.randn(2, 4, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_spiking_resnet_forward():
    model = SpikingResNet(in_channels=2, num_classes=5, block_config=[1, 1, 1, 1], num_steps=3)
    x = torch.randn(2, 3, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 5)


def test_spiking_transformer_forward():
    model = SpikingTransformer(input_dim=100, num_classes=5, d_model=32, num_heads=2, num_layers=1, num_steps=4)
    x = torch.randn(2, 4, 100)
    out = model(x)
    assert out.shape == (2, 5)


def test_encoders():
    rc = RateCoding(num_steps=10)
    data = torch.rand(4, 28, 28)
    spk_rc = rc(data)
    assert spk_rc.shape == (10, 4, 28, 28)

    tc = TemporalCoding(num_steps=8)
    spk_tc = tc(data)
    assert spk_tc.shape == (8, 4, 28, 28)

    dm = DeltaModulation(threshold=0.1)
    sig = torch.randn(5, 4, 16)
    spk_dm = dm(sig)
    assert spk_dm.shape == (5, 4, 2, 16)


def test_surrogate_gradients():
    fs = FastSigmoid(scale=20.0)
    x = torch.tensor([-0.5, 0.0, 0.5], requires_grad=True)
    out = fs(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(x.grad > 0)
