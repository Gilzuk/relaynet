import numpy as np

def ber(tx_bits, rx_bits):
    tx = np.asarray(tx_bits).ravel()
    rx = np.asarray(rx_bits).ravel()
    assert tx.shape == rx.shape
    return np.mean(tx != rx)

def symbols_to_bits(symbols):
    # assume BPSK: real>0 -> 1 else 0
    return (symbols.real > 0).astype(int)

# Example usage:
# ber_val = ber(tx_bits, symbols_to_bits(rx_symbols))
