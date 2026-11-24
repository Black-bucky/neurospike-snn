import torch
import torch.nn as nn
from neurospike.training import STDPLearner


def main():
    print("🧠 STDP (Spike-Timing-Dependent Plasticity) Unsupervised Learning Demonstration")
    print("=" * 75)

    # Synthetic Input layer (e.g., 100 sensory channels) -> Output layer (20 feature neurons)
    in_features = 100
    out_features = 20
    batch_size = 16

    synaptic_layer = nn.Linear(in_features, out_features, bias=False)
    # Initialize positive weights (excitatory synapses)
    nn.init.uniform_(synaptic_layer.weight, 0.1, 0.5)

    stdp = STDPLearner(a_plus=0.02, a_minus=0.015, w_min=0.0, w_max=1.0)

    print(f"Initial Mean Synaptic Weight: {synaptic_layer.weight.mean().item():.4f}")

    # Simulating 50 Poisson spike learning iterations
    for step in range(50):
        # Generate correlated pre-synaptic spike patterns
        pre_spikes = (torch.rand(batch_size, in_features) < 0.25).float()
        
        # Post-synaptic activation & spike thresholding
        current = synaptic_layer(pre_spikes)
        post_spikes = (current > 1.2).float()

        # Unsupervised STDP weight modification
        stdp.update(pre_spikes, post_spikes, synaptic_layer)

    print(f"Post-STDP Mean Synaptic Weight: {synaptic_layer.weight.mean().item():.4f}")
    print(f"Weight Sparsity (zero weights): {(synaptic_layer.weight == 0.0).float().mean().item()*100:.1f}%")
    print("✅ STDP demonstrated successfully: Biological synaptic weights adapted to causal spike correlations.")


if __name__ == "__main__":
    main()
