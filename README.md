# NeuroSpike

An Event-Driven Neuromorphic Spiking Neural Network Framework.

## Highlights
- **Spiking Attention & Transformers**: Multi-head spike-driven attention on Spiking Heidelberg Digits (SHD).
- **Residual SNNs & Deep CNNs**: Neuromorphic event vision for DVS-CIFAR10.
- **Biophysical Dynamics**: Adaptive LIF, Izhikevich neuron firing modes, and stochastic LIF.
- **Biologically Plausible Plasticity**: Hebbian STDP & Reward-Modulated STDP.
- **Energy Benchmark**: Direct Synaptic Operations (SynOps) vs FP32 Multiply-Accumulate (MACs) comparison.

## Quick Start

```bash
# 1. Train Spiking Transformer on neuromorphic audio
python scripts/train.py --model spiking_transformer --dataset shd

# 2. Train Spiking CNN on event camera vision
python scripts/train.py --model spiking_cnn --dataset dvs_cifar10

# 3. Demonstrate STDP unsupervised synaptic adaptation
python scripts/demo_stdp.py
```
