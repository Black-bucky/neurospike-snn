import argparse
import sys
import torch
import torch.nn as nn
from neurospike.models import SpikingCNN, SpikingResNet, SpikingTransformer
from neurospike.data import SHDDataset, DVSCIFARDataset, SpikingDataLoader
from neurospike.training import SNNTrainer
from neurospike.visualization import create_dashboard, plot_energy_comparison


def main():
    parser = argparse.ArgumentParser(description="NeuroSpike Training Script")
    parser.add_argument("--model", type=str, default="spiking_transformer", choices=["spiking_cnn", "spiking_resnet", "spiking_transformer"])
    parser.add_argument("--dataset", type=str, default="shd", choices=["shd", "dvs_cifar10"])
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_steps", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print(f"🧠 NeuroSpike: SNN Training Pipeline")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Device: {args.device}")
    print("=" * 60)

    device = torch.device(args.device)

    # Initialize Dataset
    if args.dataset == "shd":
        train_ds = SHDDataset(train=True, num_steps=args.num_steps)
        val_ds = SHDDataset(train=False, num_steps=args.num_steps)
    else:
        train_ds = DVSCIFARDataset(train=True, num_steps=args.num_steps)
        val_ds = DVSCIFARDataset(train=False, num_steps=args.num_steps)

    train_loader = SpikingDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = SpikingDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Instantiate Model
    if args.model == "spiking_transformer":
        model = SpikingTransformer(input_dim=700, num_classes=20, d_model=128, num_layers=2, num_steps=args.num_steps)
    elif args.model == "spiking_resnet":
        model = SpikingResNet(in_channels=2, num_classes=10, block_config=[1, 1, 1, 1], num_steps=args.num_steps)
    else:
        model = SpikingCNN(in_channels=2, num_classes=10, num_steps=args.num_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = SNNTrainer(model=model, optimizer=optimizer, device=device)

    # Train
    history = trainer.fit(train_loader, val_loader, num_epochs=args.num_epochs)

    # Neuromorphic Energy Efficiency Metric
    energy_report = trainer.compute_energy_efficiency(val_loader)
    print("\n⚡ Neuromorphic Energy Efficiency Report:")
    print(f"   • Synaptic Operations (SynOps): {energy_report['synaptic_operations']:.2e}")
    print(f"   • Energy Gain vs ANN (MACs): {energy_report['energy_efficiency_gain']:.2f}x less energy!")

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
