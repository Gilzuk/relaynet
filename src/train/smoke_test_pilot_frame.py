"""
Pilot-frame experiment: short frames with 1 pilot symbol followed by data symbols (BPSK).
The NN receives the full frame ([pilot, data...], complex samples encoded as real+imag) and
outputs data symbol estimates for the data positions. We compare:
 - No equalization baseline (decide on real(y))
 - Pilot-based DSP equalization (estimate h from pilot, divide)
 - NN that uses pilot implicitly (learns CSI)

Run with:
    python -m src.train.smoke_test_pilot_frame
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.data import channel_sim as cs

# Frame parameters
FRAME_LEN = 8         # total symbols per frame (pilot_len pilots + data)
PILOT_LEN = 3         # number of pilot symbols at frame start
PILOT_SYMBOL = 1.0+0j # known pilot transmitted value (complex)

# small MLP that maps flattened real+imag frame -> (FRAME_LEN-1) real outputs
class FrameMLP(nn.Module):
    def __init__(self, frame_len=FRAME_LEN, pilot_len=PILOT_LEN, hidden=128):
        super().__init__()
        input_dim = 2 * frame_len
        out_dim = frame_len - pilot_len
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x):
        return self.net(x)


def generate_frame_batch(batch_frames, frame_len=FRAME_LEN, pilot_len=PILOT_LEN):
    # returns frames (batch_frames, frame_len) complex, targets (batch_frames, frame_len-pilot_len) real BPSK
    data_sym = (2 * np.random.randint(0, 2, size=(batch_frames, frame_len - pilot_len)) - 1).astype(np.float32)
    frames = np.zeros((batch_frames, frame_len), dtype=np.complex64)
    # set pilots
    for p in range(pilot_len):
        frames[:, p] = PILOT_SYMBOL
    frames[:, pilot_len:] = data_sym.astype(np.complex64)
    return frames, data_sym


def apply_block_fade(frames):
    # frames shape (B, L)
    B, L = frames.shape
    h = (np.random.randn(B) + 1j * np.random.randn(B)) / np.sqrt(2)
    h = h.astype(np.complex64)
    # broadcast multiply
    faded = frames * h[:, None]
    return faded, h


def add_awgn(frames, snr_db):
    # cs.awgn works on flattened complex arrays; keep shape then reshape
    flat = frames.reshape(-1)
    noisy = cs.awgn(flat, snr_db)
    return noisy.reshape(frames.shape)


def train(model, epochs=8, batch_frames=1024, batches_per_epoch=40, snr_range=(0,12), device='cpu'):
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
            # prepare NN input: flatten real+imag per frame
            inp = np.concatenate([y.real, y.imag], axis=1)  # shape (B, 2*L)
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


def baseline_no_eq(snrs, n_frames=20000, device='cpu'):
    res = {}
    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        frames_faded, h = apply_block_fade(frames_tx)
        y = add_awgn(frames_faded, snr)
        # decision on real part of received data symbols (no eq)
        detected = (y[:, PILOT_LEN:].real > 0).astype(int)
        res[snr] = np.mean(data_sym != detected)
    return res


def baseline_pilot_eq(snrs, n_frames=20000, device='cpu'):
    res = {}
    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        frames_faded, h = apply_block_fade(frames_tx)
        y = add_awgn(frames_faded, snr)
        # estimate h per frame from pilots (use average over PILOT_LEN)
        pilot_rx = y[:, :PILOT_LEN]
        # divide by known pilot and average to reduce noise
        h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
        y_eq = y / h_hat[:, None]
        detected = (y_eq[:, PILOT_LEN:].real > 0).astype(int)
        res[snr] = np.mean(data_sym != detected)
    return res


def evaluate_nn(model, snrs=(0,5,10,15,20), n_frames=20000, device='cpu'):
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in snrs:
            frames_tx, data_sym = generate_frame_batch(n_frames)
            frames_faded, h = apply_block_fade(frames_tx)
            y = add_awgn(frames_faded, snr)
            inp = np.concatenate([y.real, y.imag], axis=1)
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

    print("Pilot-frame experiment (frames with 1 pilot + data)")
    model = FrameMLP()
    train(model, epochs=8, batch_frames=1024, batches_per_epoch=40, snr_range=(0,12), device=device)

    snr_list = [0, 5, 10, 15, 20]
    print("\nBaseline (no eq) BER:")
    base_no = baseline_no_eq(snr_list, n_frames=20000)
    for snr, ber in base_no.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nBaseline (pilot eq) BER:")
    base_p = baseline_pilot_eq(snr_list, n_frames=20000)
    for snr, ber in base_p.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print("\nNN-relay BER (after training):")
    nn_res = evaluate_nn(model, snrs=snr_list, n_frames=20000, device=device)
    for snr, ber in nn_res.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")


if __name__ == '__main__':
    main()
