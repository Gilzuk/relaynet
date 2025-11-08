import numpy as np

def awgn(x, snr_db):
    """Add AWGN to complex baseband signal x (numpy array of complex)"""
    snr_linear = 10**(snr_db/10.0)
    power = np.mean(np.abs(x)**2)
    noise_power = power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape))
    return x + noise

def rayleigh_fade(x, doppler=0.0):
    """Simple block Rayleigh fading (i.i.d. per sample)"""
    h = (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape))/np.sqrt(2)
    return h * x, h


def multipath_fade(x, taps=(1.0+0j, 0.5+0.0j)):
        """
        Simple frequency-selective multipath channel (finite impulse response).
        x: 1-D complex numpy array of symbols (per-frame or flattened)
        taps: sequence of complex tap gains (first tap is direct path)

        Returns (y, h_matrix)
            y: convolved output (same length as x; tail truncated to keep length)
            h_matrix: per-symbol effective channel (for each sample, the scalar multiplying the symbol)
        Note: For frame-based processing we return per-frame convolution (same shape as x input)
        This is a minimal implementation used by smoke tests.
        """
        taps = np.asarray(taps, dtype=np.complex64)
        L = taps.size
        # perform 1D convolution per sequence; for simplicity assume x is 1-D or flattened
        y = np.convolve(x, taps, mode='same')
        # For evaluation/debugging we return an effective per-sample channel approximation:
        # approximate h[n] = taps[0] (dominant path) + small contribution; here we return taps[0]
        # (caller should be aware this is a simplification for smoke tests)
        h_eff = np.full_like(x, taps[0])
        return y.astype(np.complex64), h_eff

def generate_bpsk_symbols(n):
    bits = np.random.randint(0,2,size=n)
    return 2*bits-1, bits

# Example usage:
# x, bits = generate_bpsk_symbols(1024)
# x_faded, h = rayleigh_fade(x)
# y = awgn(x_faded, snr_db=10)
