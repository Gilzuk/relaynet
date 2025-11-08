"""
Simple smoke test: generate BPSK symbols, pass through AWGN, use amplify-and-forward baseline,
and compute BER vs SNR. Self-contained (numpy only).
Run with:
    python -m src.train.smoke_test
"""
import numpy as np

def generate_bpsk(n):
    bits = np.random.randint(0, 2, size=n)
    symbols = (2 * bits - 1).astype(np.float32)  # real BPSK
    return symbols, bits

def awgn(x, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    power = np.mean(np.abs(x) ** 2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*x.shape))
    return x + noise

def detect_bpsk(rx):
    return (rx.real > 0).astype(int)

def ber(tx_bits, rx_bits):
    return np.mean(tx_bits != rx_bits)

def run_smoke_test(n_symbols=10000, snrs=(0,5,10,15,20), seed=2025):
    np.random.seed(seed)
    print(f"Smoke test: {n_symbols} symbols per SNR")
    symbols, bits = generate_bpsk(n_symbols)
    # Treat relay as amplify-and-forward: relay just forwards the noisy received samples
    for snr in snrs:
        # source -> relay
        rx_relay = awgn(symbols, snr)
        # relay forwards (AF): here we simply forward rx_relay to destination (no extra noise)
        forwarded = rx_relay
        # destination receives forwarded signal (optionally add more noise - here none)
        rx_dest = forwarded
        detected = detect_bpsk(rx_dest)
        err = ber(bits, detected)
        print(f"SNR={snr:2d} dB  BER={err:.6f}")

if __name__ == "__main__":
    run_smoke_test()
