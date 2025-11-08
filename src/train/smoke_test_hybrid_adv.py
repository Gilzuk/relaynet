"""
Advanced hybrid experiment:
 - PILOT_LEN default 4
 - Uses multipath channel (short FIR) from src.data.channel_sim.multipath_fade
 - NN predicts complex-valued corrections (outputs real+imag per data symbol)
 - Saves model checkpoint and BER vs SNR plot under eval/

Run with:
    python -m src.train.smoke_test_hybrid_adv
"""
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.data import channel_sim as cs

# Config
FRAME_LEN = 12
PILOT_LEN = 4
PILOT_SYMBOL = 1.0 + 0j
EVAL_DIR = os.path.join(os.getcwd(), 'eval')
MODEL_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# NN that refines equalized symbols: outputs real-valued symbol estimates
class ComplexRefineNN(nn.Module):
    def __init__(self, data_len=FRAME_LEN - PILOT_LEN, in_channels=6, hidden=256, dropout=0.2):
        super().__init__()
        self.data_len = data_len
        conv_channels = 64
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(conv_channels * data_len, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, data_len)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        flat = features.reshape(batch_size, -1)
        return self.head(flat)


def generate_frame_batch(batch_frames, frame_len=FRAME_LEN, pilot_len=PILOT_LEN):
    data_sym = (2 * np.random.randint(0, 2, size=(batch_frames, frame_len - pilot_len)) - 1).astype(np.float32)
    frames = np.zeros((batch_frames, frame_len), dtype=np.complex64)
    for p in range(pilot_len):
        frames[:, p] = PILOT_SYMBOL
    frames[:, pilot_len:] = data_sym.astype(np.complex64)
    return frames, data_sym


def apply_multipath(frames, taps=(1.0+0j, 0.6+0.2j, 0.2+0.1j)):
    # apply multipath per frame (convolve per-row)
    B, L = frames.shape
    out = np.zeros_like(frames)
    h_eff = np.zeros_like(frames)
    for i in range(B):
        y, h = cs.multipath_fade(frames[i, :], taps=taps)
        out[i, :] = y
        h_eff[i, :] = h
    return out, h_eff


def add_awgn(frames, snr_db):
    flat = frames.reshape(-1)
    noisy = cs.awgn(flat, snr_db)
    return noisy.reshape(frames.shape)


def build_nn_features(received_frames, h_hat):
    """Return multi-channel features for NN consumption."""
    data_len = FRAME_LEN - PILOT_LEN
    data_rx = received_frames[:, PILOT_LEN:]
    # Prevent division by zero by flooring magnitude
    safe_h = np.where(np.abs(h_hat) < 1e-6, 1e-6 + 0j, h_hat)
    y_eq = received_frames / safe_h[:, None]
    data_eq = y_eq[:, PILOT_LEN:]

    h_real = np.repeat(h_hat.real[:, None], data_len, axis=1)
    h_imag = np.repeat(h_hat.imag[:, None], data_len, axis=1)

    features = np.stack([
        data_rx.real,
        data_rx.imag,
        data_eq.real,
        data_eq.imag,
        h_real,
        h_imag,
    ], axis=1)
    return features.astype(np.float32)


@lru_cache(maxsize=None)
def _channel_matrix_cache(taps_key, frame_len):
    h = tuple(taps_key)
    H = np.zeros((frame_len, frame_len), dtype=np.complex128)
    for idx in range(frame_len):
        basis = np.zeros(frame_len, dtype=np.complex64)
        basis[idx] = 1.0 + 0.0j
        response, _ = cs.multipath_fade(basis, taps=h)
        H[:, idx] = response.astype(np.complex128)
    return H


def build_channel_matrix(channel_taps, frame_len):
    """Construct exact linear convolution matrix via cached basis responses."""
    taps_key = tuple(channel_taps)
    return _channel_matrix_cache(taps_key, frame_len)


def _sample_training_snr(snr_range):
    low, high = snr_range
    def _uniform_safe(lo, hi):
        if hi <= lo:
            return lo
        return float(np.random.uniform(lo, hi))
    draw = np.random.rand()
    if draw < 0.4:
        return _uniform_safe(max(low, -2), min(high, 4))
    if draw < 0.7:
        return _uniform_safe(max(low, 4), min(high, 10))
    if draw < 0.9:
        return _uniform_safe(max(low, 10), min(high, 16))
    return _uniform_safe(max(low, 16), min(high, 20))


def train(model, epochs=20, batch_frames=1024, batches_per_epoch=40, snr_range=(-2,18), device='cpu'):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        ep_loss = 0.0
        for _ in range(batches_per_epoch):
            snr = _sample_training_snr(snr_range)
            frames_tx, data_sym = generate_frame_batch(batch_frames)
            frames_faded, h = apply_multipath(frames_tx)
            y = add_awgn(frames_faded, snr)
            # DSP: estimate h per frame from pilots (average) and equalize
            pilot_rx = y[:, :PILOT_LEN]
            h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
            features = build_nn_features(y, h_hat)
            # target logits trained against {0,1} labels
            tgt_bits = ((data_sym + 1.0) * 0.5).astype(np.float32)
            inp_t = torch.from_numpy(features).float().to(device)
            tgt_t = torch.from_numpy(tgt_bits).float().to(device)
            logits = model(inp_t)
            loss = loss_fn(logits, tgt_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            ep_loss += loss.item()
        print(f"Epoch {ep+1}/{epochs}  avg_loss={ep_loss/batches_per_epoch:.6f}")
    # save checkpoint
    ckpt_path = os.path.join(MODEL_DIR, 'hybrid_adv_ckpt.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def apply_equalizer(rx_signal, channel_taps, snr_db=None):
    """Apply linear MMSE equalizer tailored for the FIR channel."""
    frame_len = rx_signal.shape[1]
    H = build_channel_matrix(channel_taps, frame_len)
    H_h = H.conj().T

    if snr_db is not None:
        snr_linear = 10 ** (snr_db / 10.0)
        channel_energy = np.sum(np.abs(np.asarray(channel_taps))**2)
        noise_var = channel_energy / max(snr_linear, 1e-12)
    else:
        noise_var = 1e-3  # mild regularization when SNR unknown

    lhs = H_h @ H + noise_var * np.eye(frame_len, dtype=np.complex128)
    equalizer = np.linalg.solve(lhs, H_h)

    rx_matrix = rx_signal.astype(np.complex128)
    eq_frames = (equalizer @ rx_matrix.T).T
    return eq_frames.astype(np.complex64)


def baseline_pilot_eq(snrs, n_frames=20000, channel='multipath'):
    """Pilot-based equalization with linear MMSE equalizer."""
    res = {}
    channel_taps = (1.0+0j, 0.6+0.2j, 0.2+0.1j)

    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        if channel == 'multipath':
            frames_faded, h = apply_multipath(frames_tx)
        else:
            frames_faded, h = apply_multipath(frames_tx)  # default
        y = add_awgn(frames_faded, snr)

        # Apply MMSE equalizer first
        y_eq = apply_equalizer(y, channel_taps, snr_db=snr)

        # Additional pilot-based phase correction
        pilot_rx = y_eq[:, :PILOT_LEN]
        h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
        y_eq_final = y_eq / h_hat[:, None]

        detected = (y_eq_final[:, PILOT_LEN:].real > 0).astype(int)
        detected = 2 * detected - 1  # map {0,1} to {-1,+1}
        res[snr] = np.mean(data_sym != detected)
    return res


def baseline_af(snrs, n_frames=20000, channel='multipath'):
    """Amplify-and-Forward baseline followed by linear MMSE equalizer."""
    res = {}
    channel_taps = (1.0+0j, 0.6+0.2j, 0.2+0.1j)  # Same as apply_multipath

    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        # Same channel as pilot-eq and hybrid (single multipath hop)
        if channel == 'multipath':
            frames_faded, h = apply_multipath(frames_tx)
        else:
            frames_faded, h = apply_multipath(frames_tx)
        y = add_awgn(frames_faded, snr)

        # Apply MMSE equalizer
        y_eq = apply_equalizer(y, channel_taps, snr_db=snr)

        # Direct detection on equalized signal
        detected = (y_eq[:, PILOT_LEN:].real > 0).astype(int)
        detected = 2 * detected - 1
        res[snr] = np.mean(data_sym != detected)
    return res


def evaluate_hybrid(model, snrs=(0,5,10,15,20), n_frames=20000, device='cpu'):
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in snrs:
            frames_tx, data_sym = generate_frame_batch(n_frames)
            frames_faded, h = apply_multipath(frames_tx)
            y = add_awgn(frames_faded, snr)
            pilot_rx = y[:, :PILOT_LEN]
            h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
            features = build_nn_features(y, h_hat)
            inp_t = torch.from_numpy(features).float().to(device)
            pred = model(inp_t).cpu().numpy()
            # pred is shape (N, data_len) - real-valued BPSK symbol estimates
            # decision: threshold at 0, map to {-1, +1}
            detected = (pred > 0).astype(int)
            detected = 2 * detected - 1
            results[snr] = np.mean(data_sym != detected)
    return results


def plot_and_save(baseline_res, hybrid_res, af_res=None, out_prefix='hybrid_adv'):
    snrs = sorted(baseline_res.keys())
    base = [baseline_res[s] for s in snrs]
    hyb = [hybrid_res[s] for s in snrs]
    plt.figure()
    if af_res is not None:
        af = [af_res[s] for s in snrs]
        plt.plot(snrs, af, marker='s', label='AF + MMSE eq', linestyle='--')
    plt.plot(snrs, base, marker='o', label='Pilot + MMSE eq')
    plt.plot(snrs, hyb, marker='x', label='Hybrid NN')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.title('BER vs SNR: NN vs Traditional Equalization')
    png = os.path.join(EVAL_DIR, f'{out_prefix}.png')
    plt.savefig(png)
    # save CSV
    csvp = os.path.join(EVAL_DIR, f'{out_prefix}.csv')
    with open(csvp, 'w') as f:
        if af_res is not None:
            f.write('snr,af_mmse_eq,pilot_mmse_eq,hybrid_nn\n')
            for s in snrs:
                f.write(f"{s},{af_res[s]},{baseline_res[s]},{hybrid_res[s]}\n")
        else:
            f.write('snr,pilot_mmse_eq,hybrid_nn\n')
            for s in snrs:
                f.write(f"{s},{baseline_res[s]},{hybrid_res[s]}\n")
    print(f"Saved plot to {png} and CSV to {csvp}")


def main():
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cpu'

    print('Advanced hybrid experiment: multipath + complex-output NN')
    model = ComplexRefineNN()
    ckpt = train(model, epochs=20, batch_frames=1024, batches_per_epoch=40, snr_range=(-2,18), device=device)

    snr_list = [0,5,10,15,20]
    
    print('\nAF + MMSE Equalizer BER:')
    af_res = baseline_af(snr_list, n_frames=20000)
    for snr, ber in af_res.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")
    
    print('\nPilot + MMSE Equalizer BER:')
    base_p = baseline_pilot_eq(snr_list, n_frames=20000)
    for snr, ber in base_p.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    print('\nHybrid NN BER (after training):')
    nn_res = evaluate_hybrid(model, snrs=snr_list, n_frames=20000, device=device)
    for snr, ber in nn_res.items():
        print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

    plot_and_save(base_p, nn_res, af_res=af_res, out_prefix='hybrid_adv')

if __name__ == '__main__':
    main()
