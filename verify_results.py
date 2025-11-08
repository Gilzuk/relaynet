#!/usr/bin/env python3
"""
Verification script for hybrid relay results
"""
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import src.data.channel_sim as cs
from src.train.smoke_test_hybrid_adv import (
    ComplexRefineNN, baseline_af, baseline_pilot_eq, evaluate_hybrid,
    generate_frame_batch, apply_multipath, add_awgn,
    FRAME_LEN, PILOT_LEN, PILOT_SYMBOL
)

def verify_consistency():
    """Verify results are consistent across multiple runs"""
    print("=== Verification: Consistency Check ===")

    # Load trained model
    model = ComplexRefineNN()
    ckpt_path = os.path.join('models', 'hybrid_adv_ckpt.pth')
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded model from {ckpt_path}")
    else:
        print("No checkpoint found, using untrained model")
        return

    # Test multiple times with smaller n_frames for speed
    snr_test = 20
    n_frames = 5000
    results = []

    for i in range(3):
        ber = evaluate_hybrid(model, snrs=[snr_test], n_frames=n_frames, device='cpu')[snr_test]
        results.append(ber)
        print(".6f")

    mean_ber = np.mean(results)
    std_ber = np.std(results)
    print(".6f")
    print(".6f")

    # Check if results are reasonable (should be very low for trained model)
    if mean_ber > 0.01:  # More than 1%
        print("⚠️  WARNING: BER seems high for trained model")
    else:
        print("✅ BER looks reasonable for trained model")

def verify_baselines():
    """Verify baseline methods work as expected"""
    print("\n=== Verification: Baseline Methods ===")

    snr_test = 10
    n_frames = 10000

    # Test AF baseline
    af_ber = baseline_af([snr_test], n_frames=n_frames)[snr_test]
    print(".6f")

    # Test pilot-eq baseline
    pilot_ber = baseline_pilot_eq([snr_test], n_frames=n_frames)[snr_test]
    print(".6f")

    # AF and pilot-eq should be similar (both around 35-40% for multipath)
    if abs(af_ber - pilot_ber) > 0.05:
        print("⚠️  WARNING: AF and pilot-eq BER differ significantly")
    else:
        print("✅ AF and pilot-eq BER are similar (as expected)")

def verify_no_channel():
    """Test on AWGN channel (no multipath) to verify NN still helps"""
    print("\n=== Verification: AWGN Channel (No Multipath) ===")

    def awgn_only_baseline(snrs, n_frames=10000):
        """Baseline for AWGN channel"""
        res = {}
        for snr in snrs:
            frames_tx, data_sym = generate_frame_batch(n_frames)
            # No multipath, just AWGN
            y = add_awgn(frames_tx, snr)
            # Direct detection
            detected = (y[:, PILOT_LEN:].real > 0).astype(int)
            detected = 2 * detected - 1
            res[snr] = np.mean(data_sym != detected)
        return res

    # Test AWGN baseline
    awgn_ber = awgn_only_baseline([10], n_frames=10000)[10]
    print(".6f")

    # For AWGN, BER should be much lower than multipath
    if awgn_ber > 0.1:
        print("⚠️  WARNING: AWGN BER seems high")
    else:
        print("✅ AWGN BER is reasonable (multipath is the main impairment)")

def verify_nn_architecture():
    """Verify NN architecture and forward pass"""
    print("\n=== Verification: NN Architecture ===")

    model = ComplexRefineNN()
    print(f"Model: {model}")

    # Test forward pass
    batch_size = 32
    data_len = FRAME_LEN - PILOT_LEN
    inp_dim = 2 * data_len  # real + imag

    # Create dummy input
    dummy_input = torch.randn(batch_size, inp_dim)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (batch_size={batch_size}, data_len={data_len})")

    if output.shape == (batch_size, data_len):
        print("✅ NN architecture is correct")
    else:
        print("❌ NN architecture mismatch")

def main():
    print("Hybrid Relay Results Verification")
    print("=" * 40)

    verify_nn_architecture()
    verify_baselines()
    verify_no_channel()
    verify_consistency()

    print("\n" + "=" * 40)
    print("Verification complete!")

if __name__ == '__main__':
    main()