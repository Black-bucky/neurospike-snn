import argparse
import torch
from neurospike.models import SpikingCNN, SpikingTransformer
from neurospike.data import SHDDataset, DVSCIFARDataset, SpikingDataLoader
from neurospike.training import SNNTrainer


def main():
    parser = argparse.ArgumentParser(description="NeuroSpike Evaluation Script")
    parser.add_argument("--model", type=str, default="spiking_transformer")
    parser.add_argument("--dataset", type=str, default="shd")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Running SNN Evaluation on {args.dataset}...")

    if args.dataset == "shd":
        test_ds = SHDDataset(train=False, num_steps=args.num_steps)
        model = SpikingTransformer(input_dim=700, num_classes=20, d_model=128, num_layers=2, num_steps=args.num_steps)
    else:
        test_ds = DVSCIFARDataset(train=False, num_steps=args.num_steps)
        model = SpikingCNN(in_channels=2, num_classes=10, num_steps=args.num_steps)

    test_loader = SpikingDataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = SNNTrainer(model=model, optimizer=optimizer, device=device)

    results = trainer.evaluate(test_loader)
    print(f"✅ Test Loss: {results['loss']:.4f} | Test Accuracy: {results['acc']*100:.2f}%")


if __name__ == "__main__":
    main()
