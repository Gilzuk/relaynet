"""
PyTorch smoke test: train a tiny NN relay on synthetic BPSK + AWGN data and evaluate BER vs SNR.

Run with:
    python -m src.train.smoke_test_nn

The script:
- generates BPSK symbols
- adds AWGN (domain-randomized SNR during training)
- trains a tiny MLP to map noisy received samples to transmitted symbols (regression, MSE)
- evaluates BER vs SNR using the trained NN and prints results
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# -- data / channel utils --
def generate_bpsk(n):
    bits = np.random.randint(0, 2, size=n)
    symbols = (2 * bits - 1).astype(np.float32)  # real BPSK
    return symbols, bits

def awgn_np(x, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    power = np.mean(np.abs(x) ** 2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power / 2) * np.random.randn(*x.shape)
    return x + noise

# -- tiny relay NN --
class TinyRelayNN(nn.Module):
    def __init__(self, input_dim=1, hidden=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# -- training and evaluation --
def train_relay(model, epochs=8, batch_size=4096, batches_per_epoch=50, snr_range=(0,15), device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            # random SNR per batch for domain randomization
            snr = np.random.uniform(*snr_range)
            x_np, _ = generate_bpsk(batch_size)
            y_np = awgn_np(x_np, snr)  # source->relay noisy
            inp = torch.from_numpy(y_np.reshape(-1,1)).float().to(device)
            tgt = torch.from_numpy(x_np.reshape(-1,1)).float().to(device)
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs}  avg_loss={epoch_loss/batches_per_epoch:.6f}")

def evaluate_ber(model, snr_list=(0,5,10,15,20), n_symbols=20000, device='cpu'):
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in snr_list:
            x_np, bits = generate_bpsk(n_symbols)
            y_np = awgn_np(x_np, snr)
            inp = torch.from_numpy(y_np.reshape(-1,1)).float().to(device)
            pred = model(inp).cpu().numpy().reshape(-1)
            # decision: sign(pred) -> bit
            detected = (pred > 0).astype(int)
            ber = np.mean(bits != detected)
            results[snr] = ber
    return results

def baseline_ber(snrs, n_symbols=20000):
    res = {}
    for snr in snrs:
        x_np, bits = generate_bpsk(n_symbols)
        y_np = awgn_np(x_np, snr)
        detected = (y_np > 0).astype(int)
        res[snr] = np.mean(bits != detected)
    return res

def main():
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'  # keep CPU for smoke-test compatibility

    print("Training tiny NN relay (this is a quick smoke-test; adjust epochs/data for research)")
    model = TinyRelayNN()
    train_relay(model, epochs=8, batch_size=4096, batches_per_epoch=40, snr_range=(0,12), device=device)

    snr_list = [0, 5, 10, 15, 20]
    print("\nBaseline AF (no NN) BER:")
    base = baseline_ber(snr_list, n_symbols=20000)
    for snr, ber in base.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nNN-relay BER (after training):")
    nn_res = evaluate_ber(model, snr_list, n_symbols=20000, device=device)
    for snr, ber in nn_res.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

if __name__ == "__main__":
    main()
