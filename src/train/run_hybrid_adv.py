"""
CLI runner for advanced hybrid experiment.
Usage example:
  python -m src.train.run_hybrid_adv --epochs 20 --hidden 256
"""
import argparse
import numpy as np
import torch
from src.train.smoke_test_hybrid_adv import ComplexRefineNN, train, baseline_pilot_eq, evaluate_hybrid, plot_and_save


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--batch_frames', type=int, default=1024)
    p.add_argument('--batches_per_epoch', type=int, default=40)
    p.add_argument('--snr_min', type=float, default=5.0)
    p.add_argument('--snr_max', type=float, default=18.0)
    p.add_argument('--device', type=str, default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = args.device

    print(f"Running advanced hybrid CLI: epochs={args.epochs}, hidden={args.hidden}")
    model = ComplexRefineNN(hidden=args.hidden)
    # call train defined in smoke_test_hybrid_adv
    _ = train(model, epochs=args.epochs, batch_frames=args.batch_frames, batches_per_epoch=args.batches_per_epoch, snr_range=(args.snr_min, args.snr_max), device=device)

    snr_list = [0,5,10,15,20]
    base_p = baseline_pilot_eq(snr_list, n_frames=20000)
    nn_res = evaluate_hybrid(model, snrs=snr_list, n_frames=20000, device=device)
    plot_and_save(base_p, nn_res, out_prefix='hybrid_adv_cli')

if __name__ == '__main__':
    main()
