"""Count alive SPD components using mean CI > 1e-6 threshold.

A component is "alive" if its mean CI across 1M tokens exceeds 1e-6.
Runs one model at a time via subprocess to avoid OOM accumulation.

Usage:
    python experiments/exp_041_spd_feature_splitting/alive_mean_ci.py
    python experiments/exp_041_spd_feature_splitting/alive_mean_ci.py --run 4x
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("experiments/exp_041_spd_feature_splitting/output")
OUTPUT_FILE = OUTPUT_DIR / "alive_components_mean_ci.json"

SPD_RUNS = {
    "0.5x": "goodfire/spd/s-b2b37c4e",
    "1x (jose)": "goodfire/spd/s-55ea3f9b",
    "2x": "goodfire/spd/s-266cb440",
    "4x": "goodfire/spd/s-d3834f54",
}

WORKER_SCRIPT = r"""
import sys, json, torch
from pathlib import Path
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, '.')
sys.path.insert(0, '/workspace/spd')

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
DEVICE = 'cuda'
LAYERS = [0, 1, 2, 3]
BATCH_SIZE = 4
SEQ_LEN = 512
N_TOKENS = int(1e6)
N_BATCHES = N_TOKENS // (BATCH_SIZE * SEQ_LEN)
MLP_MODULE_PATTERNS = ['h.{}.mlp.c_fc', 'h.{}.mlp.down_proj']
THRESHOLD = 1e-6

label = sys.argv[1]
run_path = sys.argv[2]

from datasets import load_dataset
from tqdm import tqdm
from analysis.collect_spd_activations import load_spd_model

dataset = load_dataset('danbraunai/pile-uncopyrighted-tok', split='train', streaming=True)
dataset = dataset.shuffle(seed=0, buffer_size=10000)
data_iter = iter(dataset)
batches = []
for _ in tqdm(range(N_BATCHES), desc='Loading batches'):
    batch_ids = []
    for _ in range(BATCH_SIZE):
        sample = next(data_iter)
        ids = sample['input_ids']
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        batch_ids.append(ids[:SEQ_LEN])
    batches.append(torch.stack(batch_ids))

print(f'Loading {label} ({run_path})...')
spd_model, _ = load_spd_model(run_path)
spd_model.to(DEVICE)

mlp_modules = []
for layer in LAYERS:
    for pattern in MLP_MODULE_PATTERNS:
        mod_name = pattern.format(layer)
        if mod_name in spd_model.module_to_c:
            mlp_modules.append(mod_name)

ci_sum = {}
for mod_name in mlp_modules:
    n_c = spd_model.module_to_c[mod_name]
    ci_sum[mod_name] = torch.zeros(n_c, dtype=torch.float64, device='cpu')

n_tokens_total = 0
for input_ids_cpu in tqdm(batches, desc=label):
    input_ids = input_ids_cpu.to(DEVICE)
    B, S = input_ids.shape
    n_tokens_total += B * S
    out = spd_model(input_ids, cache_type='input')
    ci = spd_model.calc_causal_importances(out.cache, sampling='continuous')
    for mod_name in mlp_modules:
        ci_vals = ci.lower_leaky[mod_name].reshape(-1, spd_model.module_to_c[mod_name])
        ci_sum[mod_name] += ci_vals.double().sum(dim=0).cpu()

results = {}
for mod_name in mlp_modules:
    mean_ci = ci_sum[mod_name] / n_tokens_total
    n_alive = (mean_ci > THRESHOLD).sum().item()
    n_total = spd_model.module_to_c[mod_name]
    results[mod_name] = {'alive': n_alive, 'total': n_total}
    print(f'  {mod_name}: {n_alive}/{n_total} alive ({100*n_alive/n_total:.1f}%)')

print('RESULT_JSON:' + json.dumps(results))
"""


def run_one(label: str, run_path: str) -> dict:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, label, run_path],
        capture_output=True, text=True, timeout=1200, env=env,
    )
    print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr, end="")
    for line in result.stdout.split("\n"):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    raise RuntimeError(f"No result from {label}:\n{result.stdout[-500:]}\n{result.stderr[-500:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Run only this label (e.g. '4x')")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    runs_to_do = {args.run: SPD_RUNS[args.run]} if args.run else SPD_RUNS

    for label, run_path in runs_to_do.items():
        print(f"\n{'='*60}")
        print(f"  {label} ({run_path})")
        print(f"{'='*60}")
        results = run_one(label, run_path)
        all_results[label] = results

        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved ({len(all_results)} models so far)")

    print(f"\nDone! Results in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
