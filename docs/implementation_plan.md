# Implementation Plan

Milestones:
1. Literature & repo survey (2 weeks)
2. Minimal reproducible codebase: simulator + baseline relays (2 weeks)
3. Port 1–2 GitHub NN-based relay repos; adapt & test (2–3 weeks)
4. Implement hybrid relay NN and training pipeline (3 weeks)
5. Extensive evaluation (BER/SER, throughput, latency) and ablation studies (3 weeks)
6. Quantization/pruning and SDR prototyping (optional) (4+ weeks)

Deliverables:
- Reproducible simulator and dataset
- Ported/modified GitHub implementations with license notes
- Hybrid relay NN implementation and training scripts
- Evaluation reports and scripts

{ 
## Repository layout (recommended)
- docs/                      # literature, proposals, experiment notes
- src/
  - data/                    # channel simulators, dataset generators
  - models/                  # NN and classical relay implementations
  - train/                   # training scripts and configs
  - eval/                    # evaluation scripts and metrics
- experiments/               # experiment configs and logs
- requirements.txt
- README.md

## Quick start (local CPU)
1. create venv and install:
   - python -m venv .venv
   - .venv\Scripts\activate   (Windows) or source .venv/bin/activate
   - pip install -r requirements.txt

2. Run a smoke test:
   - python -m src.train.train_relay --config experiments/example.yaml

## Minimal dependencies (examples)
- python>=3.8
- numpy, scipy
- torch (or tensorflow) pinned version in requirements.txt
- matplotlib (for plots)

## Measurable acceptance criteria
- Reproduce baseline AF/DF BER vs SNR curves within expected tolerance on small dataset.
- Ported GitHub repo runs its main example end-to-end with provided data (documented).
- Hybrid NN achieves BER improvement or latency/complexity tradeoff vs baseline in at least one scenario.
- All experiments runnable via single CLI config; seeds fixed for reproducibility.

## Risks & mitigations
- License incompatibility: check LICENSE before porting; prefer MIT/BSD/Apache.
- Non-reproducible code: pin versions, add minimal tests, and containerize if needed.
- Training instability: start with small datasets and domain randomization; use checkpointing.

## Next actions (first-week checklist)
1. Run targeted literature + GitHub search; capture 8–10 candidate repos and licenses.
2. Create repo skeleton and requirements.txt (pin torch + numpy).
3. Implement simple BPSK AWGN and Rayleigh simulators and baseline AF/DF implementations.
4. Run baseline experiments and plot BER vs SNR for validation.

}

{ 
## Targeted GitHub search — collect candidate repos

Goal: produce a short list (8–12) candidate repositories implementing neural-network-based physical-layer components (receivers, autoencoders, relays) and capture license info.

1) Recommended search queries (try each; adjust language filter):
- "neural relay" OR "nn relay" OR "neural network relay"
- "physical layer" "autoencoder" "communications" "deep learning receiver"
- "phy autoencoder" OR "end-to-end learning communications"
Append: language:python or language:cpp as preferred.

2) Quick GH-CLI (preferred) examples
- Search repos (adjust query and --limit):
  - gh search repos "neural relay in:readme language:python" --limit 50 --json name,description,url,license,stargazersCount
- If gh doesn't return license, fetch with:
  - gh repo view owner/repo --json license

3) REST API (curl) example (requires token in GITHUB_TOKEN):
  - curl -H "Authorization: token $GITHUB_TOKEN" \
    "https://api.github.com/search/repositories?q=neural+relay+physical+layer+language:python&sort=stars&order=desc&per_page=50"

4) Small helper: fetch LICENSE for a list of repositories
- Save candidate repo full names (owner/repo) to repos.txt then run the script below to fetch license metadata and URL.

```python
# filepath: c:\Users\gilzu\THESIS\docs\implementation_plan.md
# helper: fetch license metadata for repos in repos.txt (one owner/repo per line)
import os, requests, sys

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("Set GITHUB_TOKEN env var with a personal access token (no scopes required for public repos).")
    sys.exit(1)

hdr = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
with open("repos.txt") as f:
    for line in f:
        repo = line.strip()
        if not repo: continue
        url = f"https://api.github.com/repos/{repo}/license"
        r = requests.get(url, headers=hdr)
        if r.status_code == 200:
            j = r.json()
            lic = j.get("license", {}).get("name")
            lic_url = j.get("html_url")
            print(f"{repo}\tlicense: {lic}\t{lic_url}")
        else:
            # fallback: repo metadata
            url2 = f"https://api.github.com/repos/{repo}"
            r2 = requests.get(url2, headers=hdr)
            if r2.status_code == 200:
                jr = r2.json()
                lic = jr.get("license", {}).get("name")
                print(f"{repo}\tlicense: {lic}\t(license file not at /license endpoint)")
            else:
                print(f"{repo}\tERROR HTTP {r.status_code}")
```

5) Screening checklist (manual)
- Open README: does it implement NN at PHY or relay logic?
- Check examples: is there an end-to-end example or dataset?
- LICENSE: prefer MIT/BSD/Apache for easy reuse; if GPL/other copyleft, note constraints.
- Tests & requirements: can you run a small smoke test quickly?

6) Suggested outcome format (record in a markdown or CSV)
- owner/repo | stars | brief description | license (from API) | smoke-test pass (Y/N) | notes

}

{ 
## Quick deliverable: requirements + smoke-test

A minimal, reproducible environment and a self-contained smoke-test script are provided to validate the pipeline quickly.

Files added:
- c:\Users\gilzu\THESIS\requirements.txt
- c:\Users\gilzu\THESIS\src\train\smoke_test.py

How to run (local CPU):
1. Create and activate venv:
   - python -m venv .venv
   - .venv\Scripts\activate   (Windows) or source .venv/bin/activate

2. Install dependencies:
   - pip install -r requirements.txt

3. Run the smoke test:
   - python -m src.train.smoke_test
   This runs a small BPSK + AWGN -> AF baseline and prints BER vs SNR.

Notes:
- requirements.txt pins minimal libs; adjust torch version if you need GPU builds.
- The smoke-test is self-contained (numpy only) so it runs even if torch is not used.
}
