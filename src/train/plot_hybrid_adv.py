"""
Read eval/hybrid_adv.csv and create two additional plots:
 - BER bar chart (pilot-eq vs hybrid)
 - Delta plot (pilot_e - hybrid) showing improvement
Saves outputs to eval/ as PNG files.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.getcwd()
CSV = os.path.join(ROOT, 'eval', 'hybrid_adv.csv')
OUT_BAR = os.path.join(ROOT, 'eval', 'hybrid_adv_bar.png')
OUT_DELTA = os.path.join(ROOT, 'eval', 'hybrid_adv_delta.png')

snrs = []
pilot = []
hybrid = []
with open(CSV, 'r') as f:
    r = csv.DictReader(f)
    for row in r:
        snrs.append(int(row['snr']))
        pilot.append(float(row['pilot_eq']))
        hybrid.append(float(row['hybrid']))

snrs = np.array(snrs)
pilot = np.array(pilot)
hybrid = np.array(hybrid)

dx = pilot - hybrid

# Bar chart
x = np.arange(len(snrs))
width = 0.35
plt.figure(figsize=(8,4))
plt.bar(x - width/2, pilot, width, label='pilot-eq')
plt.bar(x + width/2, hybrid, width, label='hybrid')
plt.xticks(x, snrs)
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER: pilot-eq vs hybrid')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_BAR)
print(f'Saved bar chart to {OUT_BAR}')

# Delta plot
plt.figure(figsize=(8,3))
plt.plot(snrs, dx, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('BER improvement (pilot - hybrid)')
plt.title('Hybrid improvement over pilot-eq (positive = improvement)')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DELTA)
print(f'Saved delta plot to {OUT_DELTA}')

# Also print numeric summary
for s, p, h, d in zip(snrs, pilot, hybrid, dx):
    print(f'SNR={s} dB: pilot={p:.6f}, hybrid={h:.6f}, improvement={d:.6f}')
