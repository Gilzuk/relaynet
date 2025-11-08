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
import copy
import time
from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.data import channel_sim as cs
import argparse

def resolve_device(requested='auto'):
    """Resolve the torch device to use."""
    requested = (requested or 'auto').lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            print('Using CUDA')
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            print('Using MPS')
            return torch.device('mps')
        print('Using CPU (auto fallback)')
        return torch.device('cpu')
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        print('Using CUDA')
        return torch.device('cuda')
    if requested == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but not available')
        print('Using MPS')
        return torch.device('mps')
    print('Using CPU (explicit)')
    return torch.device('cpu')



# Config
FRAME_LEN = 12
PILOT_LEN = 4
PILOT_SYMBOL = 1.0 + 0j
EVAL_DIR = os.path.join(os.getcwd(), 'eval')
MODEL_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MULTIPATH_TAPS = (1.0+0j, 0.6+0.2j, 0.2+0.1j)
AWGN_TAPS = (1.0+0j,)

# NN that refines equalized symbols: outputs real-valued symbol estimates
CHANNEL_TO_ID = {
    'multipath': 0,
    'awgn': 1,
}


def save_checkpoint(model, path, metadata=None):
    payload = {'model_state': model.state_dict(), 'num_experts': getattr(model, 'num_experts', None)}
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)
    print(f'Saved checkpoint to {path}')


def load_checkpoint(model, path, device):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and 'model_state' in payload:
        state = payload['model_state']
        meta = {k: v for k, v in payload.items() if k != 'model_state'}
    else:
        state = payload
        meta = {}
    model.load_state_dict(state)
    exp_meta = meta.get('num_experts')
    if exp_meta is not None and hasattr(model, 'num_experts') and model.num_experts != exp_meta:
        print(f'Warning: checkpoint num_experts={exp_meta} differs from model num_experts={model.num_experts}')
    print(f'Loaded checkpoint from {path}')
    return meta


class ComplexRefineNN(nn.Module):
    def __init__(self, data_len=FRAME_LEN - PILOT_LEN, in_channels=8, hidden=256, dropout=0.15, num_experts=8, expert_hidden=192):
        super().__init__()
        self.data_len = data_len
        self.num_experts = num_experts
        conv_channels = 64
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        flat_dim = conv_channels * data_len
        self.shared_proj = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, data_len)
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_experts)
        )

    def forward(self, x, return_weights=False):
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        flat = features.reshape(batch_size, -1)
        shared = self.shared_proj(flat)

        gate_logits = self.gate(shared)
        weights = torch.softmax(gate_logits, dim=1)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(shared))
        expert_stack = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, data_len)

        mixed = torch.sum(weights.unsqueeze(-1) * expert_stack, dim=1)
        if return_weights:
            return mixed, weights
        return mixed

    def prune_experts(self, active_indices):
        active_indices = sorted(set(active_indices))
        if not active_indices:
            return
        if len(active_indices) == len(self.experts):
            return
        # Prune experts
        new_experts = nn.ModuleList([self.experts[i] for i in active_indices])
        self.experts = new_experts

        # Update gate last linear layer to new size
        if not isinstance(self.gate[-1], nn.Linear):
            raise RuntimeError('Expected final gate layer to be nn.Linear')
        old_linear = self.gate[-1]
        new_out = len(active_indices)
        new_linear = nn.Linear(old_linear.in_features, new_out)
        with torch.no_grad():
            new_linear.weight.copy_(old_linear.weight[active_indices])
            new_linear.bias.copy_(old_linear.bias[active_indices])
        self.gate[-1] = new_linear.to(old_linear.weight.device)
        self.num_experts = new_out


def generate_frame_batch(batch_frames, frame_len=FRAME_LEN, pilot_len=PILOT_LEN):
    data_sym = (2 * np.random.randint(0, 2, size=(batch_frames, frame_len - pilot_len)) - 1).astype(np.float32)
    frames = np.zeros((batch_frames, frame_len), dtype=np.complex64)
    for p in range(pilot_len):
        frames[:, p] = PILOT_SYMBOL
    frames[:, pilot_len:] = data_sym.astype(np.complex64)
    return frames, data_sym


def apply_multipath(frames, taps=MULTIPATH_TAPS):
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


def add_awgn_multihop(frames, snr_db, hops=1):
    out = frames.astype(np.complex64)
    for _ in range(int(max(1, hops))):
        out = add_awgn(out, snr_db)
    return out


def get_channel_taps(channel):
    if channel == 'multipath':
        return MULTIPATH_TAPS
    if channel == 'awgn':
        return AWGN_TAPS
    raise ValueError(f"Unknown channel type: {channel}")


def apply_channel(frames, channel='multipath', hops=1):
    if hops < 1:
        raise ValueError('hops must be >= 1')
    out = frames.astype(np.complex64)
    h_total = np.ones_like(out, dtype=np.complex64)
    for _ in range(int(hops)):
        if channel == 'multipath':
            out, h_step = apply_multipath(out, taps=MULTIPATH_TAPS)
        elif channel == 'awgn':
            out = out.astype(np.complex64).copy()
            h_step = np.ones_like(out, dtype=np.complex64)
        else:
            raise ValueError(f"Unknown channel type: {channel}")
        h_total *= h_step
    return out, h_total


def build_nn_features(received_frames, h_hat, channel=None):
    """Return multi-channel features for NN consumption."""
    data_len = FRAME_LEN - PILOT_LEN
    data_rx = received_frames[:, PILOT_LEN:]
    # Prevent division by zero by flooring magnitude
    safe_h = np.where(np.abs(h_hat) < 1e-6, 1e-6 + 0j, h_hat)
    y_eq = received_frames / safe_h[:, None]
    data_eq = y_eq[:, PILOT_LEN:]

    h_real = np.repeat(h_hat.real[:, None], data_len, axis=1)
    h_imag = np.repeat(h_hat.imag[:, None], data_len, axis=1)

    feature_list = [
        data_rx.real,
        data_rx.imag,
        data_eq.real,
        data_eq.imag,
        h_real,
        h_imag,
    ]

    num_channel_types = len(CHANNEL_TO_ID)
    channel_features = []
    channel_id = CHANNEL_TO_ID.get(channel, -1)
    for idx in range(num_channel_types):
        feat = np.zeros((received_frames.shape[0], data_len), dtype=np.float32)
        if channel_id == idx:
            feat[:, :] = 1.0
        channel_features.append(feat)
    feature_list.extend(channel_features)

    features = np.stack(feature_list, axis=1)
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


def train(model, epochs=20, batch_frames=1024, batches_per_epoch=40, snr_range=(-2,18), device='cpu', val_batches=10, patience=5, weight_decay=1e-4, channels=('multipath', 'awgn'), lr=1e-3, save_path=None, metadata=None, hop_choices=(1,)):
    device = torch.device(device)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_val = float('inf')
    epochs_no_improve = 0
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for _ in range(batches_per_epoch):
            snr = _sample_training_snr(snr_range)
            channel_choice = np.random.choice(channels)
            hop_choice = int(np.random.choice(hop_choices))
            frames_tx, data_sym = generate_frame_batch(batch_frames)
            frames_faded, _ = apply_channel(frames_tx, channel=channel_choice, hops=hop_choice)
            y = add_awgn_multihop(frames_faded, snr, hops=hop_choice)
            # DSP: estimate h per frame from pilots (average) and equalize
            pilot_rx = y[:, :PILOT_LEN]
            h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
            features = build_nn_features(y, h_hat, channel_choice)
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
        train_loss = ep_loss / batches_per_epoch

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _ in range(val_batches):
                snr_val = _sample_training_snr(snr_range)
                channel_choice = np.random.choice(channels)
                hop_choice = int(np.random.choice(hop_choices))
                frames_tx, data_sym = generate_frame_batch(batch_frames)
                frames_faded, _ = apply_channel(frames_tx, channel=channel_choice, hops=hop_choice)
                y = add_awgn_multihop(frames_faded, snr_val, hops=hop_choice)
                pilot_rx = y[:, :PILOT_LEN]
                h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
                features = build_nn_features(y, h_hat, channel_choice)
                tgt_bits = ((data_sym + 1.0) * 0.5).astype(np.float32)
                inp_t = torch.from_numpy(features).float().to(device)
                tgt_t = torch.from_numpy(tgt_bits).float().to(device)
                logits = model(inp_t)
                val_loss += loss_fn(logits, tgt_t).item()
        val_loss /= max(val_batches, 1)
        print(f"Epoch {ep+1}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Early stopping bookkeeping
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {ep+1} epochs. Best val_loss={best_val:.6f}")
                break
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        save_checkpoint(model, save_path, metadata=metadata)
    return save_path


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


def baseline_pilot_eq(snrs, n_frames=20000, channel='multipath', hops=1):
    """Pilot-based equalization with linear MMSE equalizer."""
    res = {}
    channel_taps = get_channel_taps(channel)

    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        frames_faded, _ = apply_channel(frames_tx, channel=channel, hops=hops)
        y = add_awgn_multihop(frames_faded, snr, hops=hops)

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


def run_full_evaluation(model, device, snr_list=None, eval_frames=100000, usage_frames=None, prune_threshold=0.03, hops=1, prune_checkpoint_prefix='hybrid_adv', experiment_tag=''):
    if snr_list is None:
        snr_list = [0, 5, 10, 15, 20]
    if usage_frames is None:
        usage_frames = min(eval_frames, 50000)

    base_prefix = prune_checkpoint_prefix or 'hybrid_adv'
    tag_suffix = f'_{experiment_tag}' if experiment_tag else ''
    channel_configs = [
        ('multipath', 'Rayleigh Multipath Channel', f'{base_prefix}_rayleigh{tag_suffix}'),
        ('awgn', 'AWGN Flat Channel', f'{base_prefix}_awgn{tag_suffix}'),
    ]

    runtime_accum = 0.0
    rayleigh_avg_usage = None

    for channel_key, channel_label, prefix in channel_configs:
        print(f'\n=== {channel_label} ===')
        eval_start = time.perf_counter()
        af_res = baseline_af(snr_list, n_frames=eval_frames, channel=channel_key, hops=hops)
        print('AF + MMSE Equalizer BER:')
        for snr, ber in af_res.items():
            print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

        base_res = baseline_pilot_eq(snr_list, n_frames=eval_frames, channel=channel_key, hops=hops)
        print('Pilot + MMSE Equalizer BER:')
        for snr, ber in base_res.items():
            print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

        nn_res = evaluate_hybrid(model, snrs=snr_list, n_frames=eval_frames, device=device, channel=channel_key, hops=hops)
        print('Hybrid NN BER (after training):')
        for snr, ber in nn_res.items():
            print(f"  SNR={snr:2d} dB   BER={ber:.6f}")

        plot_and_save(
            base_res,
            nn_res,
            af_res=af_res,
            out_prefix=prefix,
            title=f'BER vs SNR: {channel_label}'
        )

        _, _, avg_weights = analyze_expert_usage(
            model,
            snr_list,
            usage_frames,
            device,
            channel_key,
            prefix,
            hops=hops
        )
        print('Average gating weights by expert:')
        for idx, weight in enumerate(avg_weights):
            print(f'  Expert {idx}: {weight:.4f}')
        if channel_key == 'multipath':
            rayleigh_avg_usage = avg_weights
        eval_time = time.perf_counter() - eval_start
        runtime_accum += eval_time
        print(f'{channel_label} evaluation runtime: {eval_time:.2f} seconds ({eval_time/60.0:.2f} minutes)')

    if rayleigh_avg_usage is not None:
        prune_start = time.perf_counter()
        active, removed = prune_low_usage_experts(model, rayleigh_avg_usage, threshold=prune_threshold)
        pruning_prefix = f'{base_prefix}_pruning{tag_suffix}'
        plot_pruning_summary(rayleigh_avg_usage, active, removed, prune_threshold, out_prefix=pruning_prefix)
        if removed:
            print(f'Pruned experts with low Rayleigh usage: {removed}')
            print(f'Remaining experts: {active}')
            pruned_suffix = tag_suffix if base_prefix == 'hybrid_adv' else ''
            pruned_ckpt = os.path.join(MODEL_DIR, f'{base_prefix}{pruned_suffix}_pruned_ckpt.pth')
            save_checkpoint(model, pruned_ckpt, metadata={'stage': 'pruned', 'hops': hops})
            sanity_res = evaluate_hybrid(model, snrs=snr_list, n_frames=20000, device=device, channel='multipath', hops=hops)
            print('Post-prune sanity check (Rayleigh, 20k frames):')
            for snr, ber in sanity_res.items():
                print(f"  SNR={snr:2d} dB   BER={ber:.6f}")
        else:
            print('No experts pruned; all exceeded minimum usage threshold.')
        prune_time = time.perf_counter() - prune_start
        runtime_accum += prune_time
        if removed:
            print(f'Pruning + sanity evaluation runtime: {prune_time:.2f} seconds ({prune_time/60.0:.2f} minutes)')
        else:
            print(f'Pruning check runtime: {prune_time:.2f} seconds ({prune_time/60.0:.2f} minutes)')

    return runtime_accum


def baseline_af(snrs, n_frames=20000, channel='multipath', hops=1):
    """Amplify-and-Forward baseline followed by linear MMSE equalizer."""
    res = {}
    channel_taps = get_channel_taps(channel)

    for snr in snrs:
        frames_tx, data_sym = generate_frame_batch(n_frames)
        frames_faded, _ = apply_channel(frames_tx, channel=channel, hops=hops)
        y = add_awgn_multihop(frames_faded, snr, hops=hops)

        # Apply MMSE equalizer
        y_eq = apply_equalizer(y, channel_taps, snr_db=snr)

        # Direct detection on equalized signal
        detected = (y_eq[:, PILOT_LEN:].real > 0).astype(int)
        detected = 2 * detected - 1
        res[snr] = np.mean(data_sym != detected)
    return res


def evaluate_hybrid(model, snrs=(0,5,10,15,20), n_frames=20000, device='cpu', channel='multipath', hops=1):
    device = torch.device(device)
    model.to(device)
    model.eval()
    results = {}
    with torch.no_grad():
        for snr in snrs:
            frames_tx, data_sym = generate_frame_batch(n_frames)
            frames_faded, _ = apply_channel(frames_tx, channel=channel, hops=hops)
            y = add_awgn_multihop(frames_faded, snr, hops=hops)
            pilot_rx = y[:, :PILOT_LEN]
            h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
            features = build_nn_features(y, h_hat, channel)
            inp_t = torch.from_numpy(features).float().to(device)
            pred = model(inp_t).cpu().numpy()
            # pred is shape (N, data_len) - real-valued BPSK symbol estimates
            # decision: threshold at 0, map to {-1, +1}
            detected = (pred > 0).astype(int)
            detected = 2 * detected - 1
            results[snr] = np.mean(data_sym != detected)
    return results


def analyze_expert_usage(model, snrs, n_frames, device, channel, out_prefix, hops=1):
    """Compute per-expert gating usage across SNRs and generate visualization."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    snr_list = list(snrs)
    weights_per_snr = []
    total_weights = None

    with torch.no_grad():
        for snr in snr_list:
            frames_tx, _ = generate_frame_batch(n_frames)
            frames_faded, _ = apply_channel(frames_tx, channel=channel, hops=hops)
            y = add_awgn_multihop(frames_faded, snr, hops=hops)
            pilot_rx = y[:, :PILOT_LEN]
            h_hat = np.mean(pilot_rx / PILOT_SYMBOL, axis=1)
            features = build_nn_features(y, h_hat, channel)
            inp_t = torch.from_numpy(features).float().to(device)
            _, weights = model(inp_t, return_weights=True)
            weights_np = weights.cpu().numpy()
            mean_weights = weights_np.mean(axis=0)
            weights_per_snr.append(mean_weights)
            if total_weights is None:
                total_weights = mean_weights.copy()
            else:
                total_weights += mean_weights

    if not weights_per_snr:
        return snr_list, np.zeros((0, model.num_experts)), np.zeros(model.num_experts)

    weights_matrix = np.vstack(weights_per_snr)
    avg_weights = total_weights / len(weights_per_snr)

    experts = range(weights_matrix.shape[1])
    plt.figure()
    for idx in experts:
        plt.plot(snr_list, weights_matrix[:, idx], marker='o', label=f'Expert {idx}')
    plt.ylim(bottom=0.0)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Gating Probability')
    plt.title(f'Per-Expert Usage: {channel}')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', fontsize=8)
    plot_path = os.path.join(EVAL_DIR, f'{out_prefix}_experts.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved expert usage plot to {plot_path}')

    csv_path = os.path.join(EVAL_DIR, f'{out_prefix}_experts.csv')
    header = 'snr,' + ','.join([f'expert_{idx}' for idx in experts])
    csv_data = np.column_stack([snr_list, weights_matrix])
    np.savetxt(csv_path, csv_data, delimiter=',', header=header, comments='')
    print(f'Saved expert usage CSV to {csv_path}')

    return snr_list, weights_matrix, avg_weights


def prune_low_usage_experts(model, avg_weights, threshold=0.02):
    """Prune experts whose average gating weight falls below threshold."""
    if avg_weights.size == 0:
        return [], []
    active = [idx for idx, weight in enumerate(avg_weights) if weight >= threshold]
    if not active:
        active = [int(np.argmax(avg_weights))]
    removed = [idx for idx in range(len(avg_weights)) if idx not in active]
    if removed:
        model.prune_experts(active)
    return active, removed


def plot_pruning_summary(avg_weights, active, removed, threshold, out_prefix='hybrid_adv_pruning'):
    """Create a bar plot summarizing pruning decisions and export a CSV."""
    avg_weights = np.asarray(avg_weights, dtype=np.float32)
    num_experts = avg_weights.size
    if num_experts == 0:
        return

    experts = np.arange(num_experts)
    colors = ['tab:green' if idx in active else 'tab:red' for idx in experts]

    plt.figure()
    plt.bar(experts, avg_weights, color=colors)
    plt.axhline(threshold, color='black', linestyle='--', linewidth=1.0, label=f'Threshold {threshold:.2f}')
    plt.bar([], [], color='tab:green', label='Retained')
    if removed:
        plt.bar([], [], color='tab:red', label='Pruned')
    plt.xlabel('Expert index')
    plt.ylabel('Average gating probability')
    plt.title('Rayleigh pruning summary')
    plt.ylim(bottom=0.0, top=max(0.05, float(avg_weights.max()) * 1.15))
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    plot_path = os.path.join(EVAL_DIR, f'{out_prefix}.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved pruning summary plot to {plot_path}')

    csv_path = os.path.join(EVAL_DIR, f'{out_prefix}.csv')
    with open(csv_path, 'w') as f:
        f.write('expert,avg_weight,decision\n')
        for idx, weight in enumerate(avg_weights):
            decision = 'retained' if idx in active else 'pruned'
            f.write(f"{idx},{weight:.6f},{decision}\n")
    print(f'Saved pruning summary CSV to {csv_path}')


def describe_model(model):
    """Print a lightweight model summary with parameter counts and sizes."""
    print('\nModel summary:')
    header = f"{'Parameter':40s} {'Shape':20s} {'Count':>10s}"
    print(header)
    print('-' * len(header))
    total_params = 0
    trainable_params = 0
    element_size = None
    for name, param in model.named_parameters():
        numel = param.numel()
        shape = tuple(param.shape)
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        if element_size is None:
            element_size = param.element_size()
        print(f"{name:40s} {str(shape):20s} {numel:10d}")
    element_size = element_size or 4  # default to float32 size
    approx_bytes = total_params * element_size
    approx_mb = approx_bytes / (1024 ** 2)
    print('-' * len(header))
    print(f"Total parameters: {total_params:,}  |  Trainable: {trainable_params:,}")
    print(f"Approximate parameter memory (float32): {approx_mb:.2f} MB\n")


def plot_and_save(baseline_res, hybrid_res, af_res=None, out_prefix='hybrid_adv', title='BER vs SNR: NN vs Traditional Equalization'):
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
    if title:
        plt.title(title)
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

def sweep_expert_counts(expert_list=(1, 2, 4, 8), device='cpu', snrs=(0, 5, 10, 15, 20)):
    records = []
    device = torch.device(device)
    for experts in expert_list:
        model = ComplexRefineNN(num_experts=experts).to(device)
        print(f'\n=== Sweeping num_experts={experts} ===')
        train(model, device=device)
        train(
            model,
            epochs=8,
            batch_frames=1024,
            batches_per_epoch=30,
            snr_range=(10, 20),
            device=device,
            val_batches=5,
            patience=3,
            weight_decay=1e-4,
            channels=('multipath',),
            lr=5e-4
        )
        rayleigh = evaluate_hybrid(model, snrs=snrs, n_frames=20000, device=device, channel='multipath')
        records.append((experts, rayleigh))

    plt.figure()
    for experts, rayleigh in records:
        plt.plot(snrs, [rayleigh[s] for s in snrs], marker='o', label=f'Hybrid NN (experts={experts})')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.title('BER vs SNR (Rayleigh) – expert count sweep')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    out_prefix = os.path.join(EVAL_DIR, 'hybrid_adv_rayleigh_expert_sweep')
    plt.savefig(f'{out_prefix}.png', dpi=200, bbox_inches='tight')
    plt.close()
    with open(f'{out_prefix}.csv', 'w') as f:
        header = 'snr,' + ','.join([f'exp_{s}' for s in snrs])
        f.write('experts,' + header + '\n')
        for experts, rayleigh in records:
            row = ','.join([f'{rayleigh[s]}' for s in snrs])
            f.write(f'{experts},{row}\n')
    print(f'Saved expert sweep plot/CSV to {out_prefix}.png/.csv')


def compare_multi_hop(model, device, hops_list=(1, 2, 3, 4), snrs=(0, 5, 10, 15, 20), channel='multipath', eval_frames=50000, output_prefix='hybrid_adv', experiment_tag=''):
    device = torch.device(device)
    hops_list = [int(h) for h in hops_list]
    hybrid_records = {}
    pilot_records = {}
    base_prefix = output_prefix or 'hybrid_adv'
    tag_suffix = f'_{experiment_tag}' if experiment_tag else ''

    for hops in hops_list:
        print(f'\n--- Multi-hop evaluation: hops={hops} ---')
        pilot_res = baseline_pilot_eq(snrs, n_frames=eval_frames, channel=channel, hops=hops)
        hybrid_res = evaluate_hybrid(model, snrs=snrs, n_frames=eval_frames, device=device, channel=channel, hops=hops)
        pilot_records[hops] = pilot_res
        hybrid_records[hops] = hybrid_res
        print('Pilot + MMSE Equalizer BER:')
        for snr, ber in pilot_res.items():
            print(f"  hops={hops}  SNR={snr:2d} dB   BER={ber:.6f}")
        print('Hybrid NN BER:')
        for snr, ber in hybrid_res.items():
            print(f"  hops={hops}  SNR={snr:2d} dB   BER={ber:.6f}")

    cmap = plt.get_cmap('tab10')
    plt.figure()
    for idx, hops in enumerate(hops_list):
        color = cmap(idx % cmap.N)
        pilot_vals = [pilot_records[hops][snr] for snr in snrs]
        hybrid_vals = [hybrid_records[hops][snr] for snr in snrs]
        plt.plot(snrs, pilot_vals, linestyle='--', marker='s', color=color, label=f'Pilot eq (hops={hops})')
        plt.plot(snrs, hybrid_vals, linestyle='-', marker='o', color=color, label=f'Hybrid NN (hops={hops})')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.grid(True, which='both', alpha=0.3)
    plt.title(f'BER vs SNR with Multi-hop ({channel})')
    plt.legend(ncol=2, fontsize=8)
    out_prefix = os.path.join(EVAL_DIR, f'{base_prefix}_{channel}_multihop{tag_suffix}')
    plt.savefig(f'{out_prefix}.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved multi-hop comparison plot to {out_prefix}.png')

    header_cols = ['snr'] + [f'pilot_hops{h}' for h in hops_list] + [f'hybrid_hops{h}' for h in hops_list]
    csv_path = f'{out_prefix}.csv'
    with open(csv_path, 'w') as f:
        f.write(','.join(header_cols) + '\n')
        for snr in snrs:
            row = [f'{snr}']
            row.extend(f'{pilot_records[h][snr]}' for h in hops_list)
            row.extend(f'{hybrid_records[h][snr]}' for h in hops_list)
            f.write(','.join(row) + '\n')
    print(f'Saved multi-hop comparison CSV to {csv_path}')

def _resolve_checkpoint_paths(prefix, target_hop):
    if prefix == 'hybrid_adv' and target_hop == 1:
        base = os.path.join(MODEL_DIR, 'hybrid_adv_ckpt.pth')
        fine = os.path.join(MODEL_DIR, 'hybrid_adv_finetuned_ckpt.pth')
        target = os.path.join(MODEL_DIR, 'hybrid_adv_singlehop_ckpt.pth')
    else:
        base = os.path.join(MODEL_DIR, f'{prefix}_ckpt.pth')
        fine = os.path.join(MODEL_DIR, f'{prefix}_finetuned_ckpt.pth')
        target = None
        if target_hop is not None:
            target = os.path.join(MODEL_DIR, f'{prefix}_targethop{target_hop}_ckpt.pth')
    return base, fine, target


def run_training_pipeline(
    model,
    device,
    hop_choices,
    prune_threshold,
    skip_singlehop_finetune,
    skip_hop_compare,
    hop_compare,
    load_from,
    skip_training,
    checkpoint_prefix,
    target_hop,
    eval_hops,
    experiment_tag='',
):
    device = torch.device(device)
    hop_choices = tuple(int(h) for h in hop_choices)
    model.to(device)
    describe_model(model)

    total_runtime = 0.0
    overall_start = time.perf_counter()

    base_ckpt, fine_ckpt, target_ckpt = _resolve_checkpoint_paths(checkpoint_prefix, target_hop)

    if load_from:
        if not os.path.exists(load_from):
            raise FileNotFoundError(f'Checkpoint not found: {load_from}')
        load_checkpoint(model, load_from, device)
        print('Loaded checkpoint; skipping training phase.')
    elif skip_training:
        fallback_candidates = [p for p in (target_ckpt, fine_ckpt, base_ckpt) if p]
        fallback = next((p for p in fallback_candidates if os.path.exists(p)), None)
        if fallback:
            load_checkpoint(model, fallback, device)
            print(f'Loaded default checkpoint from {fallback}; training skipped.')
        else:
            raise RuntimeError('skip_training requested but no checkpoint available. Provide --load path or run training once.')
    else:
        start_time = time.perf_counter()
        train(
            model,
            epochs=20,
            batch_frames=1024,
            batches_per_epoch=40,
            snr_range=(-2, 18),
            device=device,
            channels=('multipath', 'awgn'),
            save_path=base_ckpt,
            metadata={'stage': 'base_train', 'hop_choices': hop_choices},
            hop_choices=hop_choices
        )
        base_train_time = time.perf_counter() - start_time
        total_runtime += base_train_time
        print(f'Initial training runtime: {base_train_time:.2f} seconds ({base_train_time/60.0:.2f} minutes)')

        print('\nFine-tuning on high-SNR Rayleigh frames...')
        start_time = time.perf_counter()
        train(
            model,
            epochs=8,
            batch_frames=1024,
            batches_per_epoch=30,
            snr_range=(10, 20),
            device=device,
            val_batches=5,
            patience=3,
            weight_decay=1e-4,
            channels=('multipath',),
            lr=5e-4,
            save_path=fine_ckpt,
            metadata={'stage': 'fine_tune', 'hop_choices': hop_choices},
            hop_choices=hop_choices
        )
        fine_tune_time = time.perf_counter() - start_time
        total_runtime += fine_tune_time
        print(f'Fine-tuning runtime: {fine_tune_time:.2f} seconds ({fine_tune_time/60.0:.2f} minutes)')

    if not skip_singlehop_finetune and target_hop is not None:
        if target_hop == 1:
            print('\nTargeted single-hop fine-tune...')
        else:
            print(f'\nTargeted hop {target_hop} fine-tune...')
        start_time = time.perf_counter()
        train(
            model,
            epochs=6,
            batch_frames=1024,
            batches_per_epoch=30,
            snr_range=(0, 16),
            device=device,
            val_batches=5,
            patience=3,
            weight_decay=5e-5,
            channels=('multipath',),
            lr=3e-4,
            save_path=target_ckpt,
            metadata={'stage': 'target_hop_finetune', 'target_hop': target_hop},
            hop_choices=(target_hop,)
        )
        target_time = time.perf_counter() - start_time
        total_runtime += target_time
        print(f'Targeted fine-tuning runtime: {target_time:.2f} seconds ({target_time/60.0:.2f} minutes)')

    eval_runtime = run_full_evaluation(
        model,
        device=device,
        prune_threshold=prune_threshold,
        hops=eval_hops,
        prune_checkpoint_prefix=checkpoint_prefix,
        experiment_tag=experiment_tag
    )
    total_runtime += eval_runtime

    aggregate = time.perf_counter() - overall_start
    label = f'[{checkpoint_prefix}] ' if checkpoint_prefix != 'hybrid_adv' or experiment_tag else ''
    print(f'{label}Aggregate runtime for this run (profiling sum): {total_runtime:.2f} seconds ({total_runtime/60.0:.2f} minutes)')
    print(f'{label}Wall-clock runtime: {aggregate:.2f} seconds ({aggregate/60.0:.2f} minutes)')

    if not skip_hop_compare:
        hop_list = hop_compare or (1, 2, 3, 4)
        compare_multi_hop(model, device=device, hops_list=hop_list, output_prefix=checkpoint_prefix, experiment_tag=experiment_tag)

    return model


def main(device='cpu', load_from=None, skip_training=False, prune_threshold=0.03, hop_compare=None, skip_hop_compare=False, train_hop_choices=(1,), skip_singlehop_finetune=False, train_each_hop=False):
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    print('Advanced hybrid experiment: multipath + complex-output NN')

    hop_tuple = tuple(int(h) for h in train_hop_choices) if train_hop_choices else (1,)

    if train_each_hop:
        if load_from:
            raise ValueError('Cannot use --load with --train-each-hop; provide hop-specific checkpoints manually if needed.')
        models = {}
        for hop in sorted(set(hop_tuple)):
            print(f'\n=== Dedicated training for hop count {hop} ===')
            hop_model = ComplexRefineNN().to(device)
            models[hop] = run_training_pipeline(
                hop_model,
                device=device,
                hop_choices=(hop,),
                prune_threshold=prune_threshold,
                skip_singlehop_finetune=skip_singlehop_finetune,
                skip_hop_compare=True,
                hop_compare=(hop,),
                load_from=None,
                skip_training=skip_training,
                checkpoint_prefix=f'hybrid_adv_hop{hop}',
                target_hop=hop,
                eval_hops=hop,
                experiment_tag=''
            )
        return models

    model = ComplexRefineNN().to(device)
    return run_training_pipeline(
        model,
        device=device,
        hop_choices=hop_tuple,
        prune_threshold=prune_threshold,
        skip_singlehop_finetune=skip_singlehop_finetune,
        skip_hop_compare=skip_hop_compare,
        hop_compare=hop_compare,
        load_from=load_from,
        skip_training=skip_training,
        checkpoint_prefix='hybrid_adv',
        target_hop=1,
        eval_hops=1,
        experiment_tag=''
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced hybrid experiment (Rayleigh/AWGN)')
    parser.add_argument('--device', default='auto', help='cpu | cuda | mps | auto (default)')
    parser.add_argument('--load', help='Path to checkpoint to load instead of training')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and reuse latest checkpoint')
    parser.add_argument('--skip-singlehop-finetune', action='store_true', help='Skip the targeted single-hop fine-tuning stage')
    parser.add_argument('--skip-sweep', action='store_true', help='Skip the expert-count sweep (faster)')
    parser.add_argument('--skip-hop-compare', action='store_true', help='Skip multi-hop comparison (1-4 hops)')
    parser.add_argument('--hop-list', default='1,2,3,4', help='Comma-separated hop counts for multi-hop comparison (default: 1,2,3,4)')
    parser.add_argument('--train-hop-list', default='1', help='Comma-separated hop counts sampled during training (default: 1)')
    parser.add_argument('--prune-threshold', type=float, default=0.03, help='Minimum gating weight to keep expert active')
    parser.add_argument('--train-each-hop', action='store_true', help='Train a separate MOE for each hop count in --train-hop-list')
    args = parser.parse_args()
    device = resolve_device(args.device)
    hop_list = tuple(int(h.strip()) for h in args.hop_list.split(',') if h.strip()) if args.hop_list else (1,)
    train_hop_list = tuple(int(h.strip()) for h in args.train_hop_list.split(',') if h.strip()) if args.train_hop_list else (1,)
    main(
        device=device,
        load_from=args.load,
        skip_training=args.skip_training,
        prune_threshold=args.prune_threshold,
        hop_compare=hop_list,
        skip_hop_compare=args.skip_hop_compare,
        train_hop_choices=train_hop_list,
        skip_singlehop_finetune=args.skip_singlehop_finetune,
        train_each_hop=args.train_each_hop
    )
    if not args.skip_sweep:
        sweep_expert_counts(device=device)
