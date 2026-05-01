"""Single source of truth for the wandb / VPD identifiers used by the paper.

Every experiment script imports from this module instead of hardcoding wandb
paths. To reproduce against a different set of checkpoints, edit the values
here in one place rather than hunting through every script.

Note on project names: training scripts call `wandb.init(project=<bare>)` and
require the bare project name; eval scripts call `wandb.Api().runs("<entity>/<project>")`
and require the entity-qualified form. We expose both via the bare strings
plus a `qualified()` helper.

(Wandb run paths and project names contain the strings "jose" / "spd" —
these are baked into the cloud and remain unchanged. Everything else uses
the paper terminology: `LLM_BASE_MODEL` for the target transformer and VPD
for the sparse parameter decomposition method.)
"""

from pathlib import Path

# The 4-layer LlamaSimpleMLP target model (n_embd=768, GELU, Pile-trained).
LLM_BASE_MODEL = "goodfire/spd/runs/t-9d2b8f02"

# Local cache for the base model. Training scripts auto-download here on
# first run; the eval scripts read from the same path. Path is relative to
# the repo root (where Python is invoked from).
LLM_BASE_MODEL_CACHE = Path("experiments/llm_base_model")

# Four VPD checkpoints: same training recipe, log2 capacity sweep.
VPD_CAPACITY_RUNS = {
    "0.5x": "goodfire/spd/s-b2b37c4e",
    "1x":   "goodfire/spd/s-55ea3f9b",
    "2x":   "goodfire/spd/s-266cb440",
    "4x":   "goodfire/spd/s-d3834f54",
}

# The 1x VPD model is the canonical baseline whenever a single VPD reference
# is needed (e.g. to recover the base target model from its `target_model`
# attribute).
VPD_BASELINE_RUN = VPD_CAPACITY_RUNS["1x"]

# WandB entity that owns the paper's training projects.
WANDB_ENTITY = "mats-sprint"

# Bare wandb project names (no entity prefix). Use `qualified(name)` to add
# the entity for `wandb.Api().runs(...)`.
PROJECT_LOCAL_4K_NAME  = "pile_local_sweep_jose"
PROJECT_LOCAL_32K_NAME = "pile_local_sweep_jose_32k"
PROJECT_E2E_4K_NAME    = "pile_e2e_sweep_jose"
PROJECT_E2E_32K_NAME   = "pile_e2e_sweep_jose_32k"


def qualified(project_name: str) -> str:
    """Prefix a bare project name with the wandb entity (`mats-sprint/foo`)."""
    return f"{WANDB_ENTITY}/{project_name}"


# Convenience: entity-qualified project names for `wandb.Api().runs(...)` calls.
PROJECT_LOCAL_4K  = qualified(PROJECT_LOCAL_4K_NAME)
PROJECT_LOCAL_32K = qualified(PROJECT_LOCAL_32K_NAME)
PROJECT_E2E_4K    = qualified(PROJECT_E2E_4K_NAME)
PROJECT_E2E_32K   = qualified(PROJECT_E2E_32K_NAME)

# dict_size -> wandb project, used by the training scripts to choose where
# to log based on a `--dict_size` CLI flag. Bare names (no entity prefix).
LOCAL_PROJECT_BY_DICT_SIZE = {4096: PROJECT_LOCAL_4K_NAME, 32768: PROJECT_LOCAL_32K_NAME}
E2E_PROJECT_BY_DICT_SIZE   = {4096: PROJECT_E2E_4K_NAME,   32768: PROJECT_E2E_32K_NAME}

# Hand-picked PLT and CLT runs used in the headline figures (k=16 in both
# cases). Format: dict_size -> (entity_qualified_project, wandb_run_id).
HEADLINE_TC_RUNS = {
    4096:  (PROJECT_LOCAL_4K,  "4ziu27fn"),
    32768: (PROJECT_LOCAL_32K, "c4o8i98k"),
}
HEADLINE_CLT_RUNS = {
    4096:  (PROJECT_LOCAL_4K,  "77sgz1pe"),
    32768: (PROJECT_LOCAL_32K, "j20m9hzr"),
}
