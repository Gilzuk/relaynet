"""
Runner for scaled hybrid experiment (DSP equalize + NN refine).
Usage:
    python -m src.train.run_hybrid_scaled --epochs 30 --hidden 256 --batch_frames 1024 --batches_per_epoch 60
"""
import argparse
import numpy as np
import torch
from src.train.smoke_test_hybrid import RefineNN, train, baseline_no_eq, baseline_pilot_eq, evaluate_hybrid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch_frames", type=int, default=1024)
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
    device = args.device

    print(f"Scaled hybrid run: epochs={args.epochs}, hidden={args.hidden}, batch_frames={args.batch_frames}, batches_per_epoch={args.batches_per_epoch}, snr_range=({args.snr_min},{args.snr_max}), device={device}")
    model = RefineNN(hidden=args.hidden)
    # call smoke_test_hybrid.train with the chosen params
    train(model, epochs=args.epochs, batch_frames=args.batch_frames, batches_per_epoch=args.batches_per_epoch, snr_range=(args.snr_min, args.snr_max), device=device)

    snr_list = [0, 5, 10, 15, 20]
    print("\nBaseline (no eq) BER:")
    base_no = baseline_no_eq(snr_list, n_frames=20000)
    for snr, ber in base_no.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nBaseline (pilot eq) BER:")
    base_p = baseline_pilot_eq(snr_list, n_frames=20000)
    for snr, ber in base_p.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nHybrid DSP+NN BER (after training):")
    nn_res = evaluate_hybrid(model, snrs=snr_list, n_frames=20000, device=device)
    for snr, ber in nn_res.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")


if __name__ == '__main__':
    main()
