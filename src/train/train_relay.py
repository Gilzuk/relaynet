import torch
import torch.nn as nn
import torch.optim as optim
from ..models.hybrid_relay import CompactRelayNN
from ..data import channel_sim as cs
import numpy as np

def train_epoch(model, optimizer, batch_size=256, snr_range=(0,20), device='cpu'):
    model.train()
    loss_fn = nn.MSELoss()  # placeholder; could be BER-aware loss
    for _ in range(100):  # mini-batches per epoch (adjust)
        # generate synthetic data
        x_real, bits = cs.generate_bpsk_symbols(batch_size)
        # convert to complex and pass through channel
        x = x_real.astype(np.complex64)
        snr = np.random.uniform(*snr_range)
        y = cs.awgn(x, snr)
        # prepare inputs for NN (real-imag interleaved)
        xy = np.stack([y.real, y.imag], axis=1)
        tx = np.stack([x.real, x.imag], axis=1)
        inp = torch.from_numpy(xy).float().to(device)
        tgt = torch.from_numpy(tx).float().to(device)
        pred = model(inp)
        loss = loss_fn(pred, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    device = 'cpu'
    model = CompactRelayNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        train_epoch(model, optimizer, device=device)
        # ...existing code...
