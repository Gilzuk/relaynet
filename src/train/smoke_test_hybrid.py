"""
Hybrid DSP+NN experiment: use pilot symbols to estimate channel (DSP), equalize, then train an NN
that refines equalized data symbols. We compare:
 - No equalization baseline
 - Pilot-based DSP equalization
 - Hybrid DSP + NN refinement

Run with:
    python -m src.train.smoke_test_hybrid
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import channel_sim as cs

# Frame / pilot config (keep consistent with pilot experiment)
FRAME_LEN = 8
PILOT_LEN = 3
PILOT_SYMBOL = 1.0 + 0j

# NN that refines equalized data symbols per frame
class RefineNN(nn.Module):
    def __init__(self, data_len=FRAME_LEN - PILOT_LEN, hidden=128):
        super().__init__()
        input_dim = 2 * data_len
        output_dim = data_len
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


def generate_frame_batch(batch_frames, frame_len=FRAME_LEN, pilot_len=PILOT_LEN):
    data_sym = (2 * np.random.randint(0, 2, size=(batch_frames, frame_len - pilot_len)) - 1).astype(np.float32)
    frames = np.zeros((batch_frames, frame_len), dtype=np.complex64)
    for p in range(pilot_len):
        frames[:, p] = PILOT_SYMBOL
    frames[:, pilot_len:] = data_sym.astype(np.complex64)
    return frames, data_sym


def apply_block_fade(frames):
    B, L = frames.shape
    h = (np.random.randn(B) + 1j * np.random.randn(B)) / np.sqrt(2)
    h = h.astype(np.complex64)
    faded = frames * h[:, None]
    return faded, h


def add_awgn(frames, snr_db):
    flat = frames.reshape(-1)
    noisy = cs.awgn(flat, snr_db)
    return noisy.reshape(frames.shape)


def train(model, epochs=12, batch_frames=1024, batches_per_epoch=40, snr_range=(0,12), device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        ep_loss = 0.0
        for _ in range(batches_per_epoch):
            snr = np.random.uniform(*snr_range)
            frames_tx, data_sym = generate_frame_batch(batch_frames)
            frames_faded, h = apply_block_fade(frames_tx)
            y = add_awgn(frames_faded, snr)
            # DSP: estimate h per frame from pilots and equalize
            pilot_rx = y[:, :PILOT_LEN]
            h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
            y_eq = y / h_hat[:, None]
            # prepare NN input: equalized data symbols (real+imag)
            data_eq = y_eq[:, PILOT_LEN:]
            inp = np.concatenate([data_eq.real, data_eq.imag], axis=1)
            tgt = data_sym.reshape(batch_frames, -1)
            inp_t = torch.from_numpy(inp).float().to(device)
            tgt_t = torch.from_numpy(tgt).float().to(device)
            pred = model(inp_t)
            loss = loss_fn(pred, tgt_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs}  avg_loss={ep_loss/batches_per_epoch:.6f}")


def baseline_no_eq(snrs, n_frames=20000):
    res = {}
    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        frames_faded, h = apply_block_fade(frames_tx)
        y = add_awgn(frames_faded, snr)
        detected = (y[:, PILOT_LEN:].real > 0).astype(int)
        res[snr] = np.mean(data_sym != detected)
    return res


def baseline_pilot_eq(snrs, n_frames=20000):
    res = {}
    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        frames_faded, h = apply_block_fade(frames_tx)
        y = add_awgn(frames_faded, snr)
        pilot_rx = y[:, :PILOT_LEN]
        h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
        y_eq = y / h_hat[:, None]
        detected = (y_eq[:, PILOT_LEN:].real > 0).astype(int)
        res[snr] = np.mean(data_sym != detected)
    return res


def evaluate_hybrid(model, snrs=(0,5,10,15,20), n_frames=20000, device='cpu'):
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in snrs:
            frames_tx, data_sym = generate_frame_batch(n_frames)
            frames_faded, h = apply_block_fade(frames_tx)
            y = add_awgn(frames_faded, snr)
            # DSP equalize
            pilot_rx = y[:, :PILOT_LEN]
            h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
            y_eq = y / h_hat[:, None]
            data_eq = y_eq[:, PILOT_LEN:]
            inp = np.concatenate([data_eq.real, data_eq.imag], axis=1)
            inp_t = torch.from_numpy(inp).float().to(device)
            pred = model(inp_t).cpu().numpy()
            detected = (pred > 0).astype(int)
            results[snr] = np.mean(data_sym != detected)
    return results


def main():
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'

    print("Hybrid DSP+NN experiment: pilot DSP equalize -> NN refine")
    model = RefineNN()
    train(model, epochs=12, batch_frames=1024, batches_per_epoch=40, snr_range=(0,12), device=device)

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
