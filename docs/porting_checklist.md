# Porting Checklist

1. License & attribution:
   - Inspect LICENSE file; ensure compatible reuse.
   - Retain original copyright notices.

2. Environment:
   - Capture requirements (requirements.txt / conda).
   - Pin versions where possible.

3. Tests:
   - Run unit/integration tests; add minimal tests if missing.

4. Modularization:
   - Separate channel sim, dataset, model, trainer, eval.
   - Replace hard-coded paths; add config file.

5. Reproducibility:
   - Fix random seeds, document training hyperparams, provide sample scripts.

6. Validation:
   - Reproduce reported results on a small subset before adaptation.
