"""
Small CLI wrapper to run the smoke-test trainer with adjustable parameters.
Run with:
    python -m src.train.run_train_scaled --epochs 20 --batch_size 4096 --batches_per_epoch 60
"""
import argparse
import numpy as np
import torch
from .smoke_test_nn import TinyRelayNN, train_relay, evaluate_ber, baseline_ber


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--batches_per_epoch", type=int, default=60)
    p.add_argument("--snr_min", type=float, default=0.0)
    p.add_argument("--snr_max", type=float, default=12.0)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Running scaled training: epochs={args.epochs}, batch_size={args.batch_size}, batches_per_epoch={args.batches_per_epoch}, snr_range=({args.snr_min},{args.snr_max}), device={args.device}")
    model = TinyRelayNN()
    train_relay(model, epochs=args.epochs, batch_size=args.batch_size, batches_per_epoch=args.batches_per_epoch, snr_range=(args.snr_min, args.snr_max), device=args.device)

    snr_list = [0, 5, 10, 15, 20]
    print("\nBaseline AF (no NN) BER:")
    base = baseline_ber(snr_list, n_symbols=20000)
    for snr, ber in base.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nNN-relay BER (after training):")
    nn_res = evaluate_ber(model, snr_list, n_symbols=20000, device=args.device)
    for snr, ber in nn_res.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")


if __name__ == "__main__":
    main()
