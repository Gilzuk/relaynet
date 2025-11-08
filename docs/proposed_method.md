# Proposed Method — Hybrid Relay with Online Adaptation

Core ideas:
1. Hybrid design: combine classical signal-processing front-end (synchronization, coarse equalization) with a compact NN relay block that performs refinement/forwarding.
2. Train end-to-end with domain randomization (SNR, fading, delays) to improve generalization.
3. Use lightweight model (depthwise convs / small MLP) and quantization-aware training for low-latency deployment.
4. Online adaptation:
   - Periodic fine-tuning at relay using small buffers and self-supervision (e.g., reconstructive or cycle-consistency losses).
   - Optionally federated updates across relays to share improvements while preserving bandwidth/privacy.
5. Robustness:
   - Adversarial domain augmentation (impaired carriers, non-Gaussian noise).
   - Out-of-distribution detection to fall back to classical AF/DF when NN confidence is low.

Evaluation:
- Metrics: BER, SER, throughput, CPU latency, model size, power.
- Baselines: Amplify-and-forward (AF), Decode-and-forward (DF), MMSE relays.
