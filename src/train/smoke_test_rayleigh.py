"""
Train a tiny NN to recover BPSK symbols over a Rayleigh-faded AWGN channel.
Compares three approaches:
 - No equalization baseline: decision = sign(real(y))
 - Ideal equalization baseline: y / h -> decision = sign(real)
 - NN: small MLP mapping [real(y), imag(y)] -> transmitted symbol (real)

Run with:
    python -m src.train.smoke_test_rayleigh
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import channel_sim as cs


def generate_bpsk(n):
    bits = np.random.randint(0, 2, size=n)
    symbols = (2 * bits - 1).astype(np.float32)  # real BPSK
    return symbols, bits


def awgn(x, snr_db):
    return cs.awgn(x, snr_db)


def rayleigh_fade(x):
    return cs.rayleigh_fade(x)


class ComplexTinyNN(nn.Module):
    def __init__(self, input_dim=2, hidden=64, output_dim=1):
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


def train(model, epochs=12, batch_size=2048, batches_per_epoch=80, snr_range=(0,12), device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        ep_loss = 0.0
        for _ in range(batches_per_epoch):
            snr = np.random.uniform(*snr_range)
            x_np, _ = generate_bpsk(batch_size)
            x_c = x_np.astype(np.complex64)
            x_faded, h = rayleigh_fade(x_c)
            y = awgn(x_faded, snr)
            inp = np.stack([y.real, y.imag], axis=1)
            tgt = x_np.reshape(-1,1)
            inp_t = torch.from_numpy(inp).float().to(device)
            tgt_t = torch.from_numpy(tgt).float().to(device)
            pred = model(inp_t)
            loss = loss_fn(pred, tgt_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs}  avg_loss={ep_loss/batches_per_epoch:.6f}")


def baseline_no_eq(snrs, n_symbols=20000):
    res = {}
    for snr in snrs:
        x_np, bits = generate_bpsk(n_symbols)
        x_c = x_np.astype(np.complex64)
        x_faded, h = rayleigh_fade(x_c)
        y = awgn(x_faded, snr)
        detected = (y.real > 0).astype(int)
        res[snr] = np.mean(bits != detected)
    return res


def baseline_ideal_eq(snrs, n_symbols=20000):
    res = {}
    for snr in snrs:
        x_np, bits = generate_bpsk(n_symbols)
        x_c = x_np.astype(np.complex64)
        x_faded, h = rayleigh_fade(x_c)
        y = awgn(x_faded, snr)
        # ideal equalization (assume receiver knows h)
        y_eq = y / h
        detected = (y_eq.real > 0).astype(int)
        res[snr] = np.mean(bits != detected)
    return res


def evaluate_nn(model, snrs=(0,5,10,15,20), n_symbols=20000, device='cpu'):
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in snrs:
            x_np, bits = generate_bpsk(n_symbols)
            x_c = x_np.astype(np.complex64)
            x_faded, h = rayleigh_fade(x_c)
            y = awgn(x_faded, snr)
            inp = np.stack([y.real, y.imag], axis=1)
            inp_t = torch.from_numpy(inp).float().to(device)
            pred = model(inp_t).cpu().numpy().reshape(-1)
            detected = (pred > 0).astype(int)
            results[snr] = np.mean(bits != detected)
    return results


def main():
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'

    print("Training NN on Rayleigh+AWGN (quick experiment)")
    model = ComplexTinyNN()
    train(model, epochs=12, batch_size=2048, batches_per_epoch=80, snr_range=(0,12), device=device)

    snr_list = [0, 5, 10, 15, 20]
    print("\nBaseline (no eq) BER:")
    base_no = baseline_no_eq(snr_list, n_symbols=20000)
    for snr, ber in base_no.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nBaseline (ideal eq) BER:")
    base_ie = baseline_ideal_eq(snr_list, n_symbols=20000)
    for snr, ber in base_ie.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nNN-relay BER (after training):")
    nn_res = evaluate_nn(model, snrs=snr_list, n_symbols=20000, device=device)
    for snr, ber in nn_res.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")


if __name__ == '__main__':
    main()
