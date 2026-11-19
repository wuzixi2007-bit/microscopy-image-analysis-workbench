"""Central paths/config so model wrappers and scripts agree on where checkpoints live."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "data" / "outputs"

# SAM
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"

DEVICE = "cuda"  # resolves to "mps" here (no cuda on Apple Silicon) via resolve_device().
# Re-benchmarked on the new machine (Apple M5 Pro) rather than assumed: SAM image encode averaged
# 0.33s/image on MPS vs 1.30s/image on CPU (3-run average after warmup) -- MPS is ~4x faster here,
# the opposite of the old Intel Mac's result (MPS was SLOWER than CPU there, 21.7s vs 9.5s, because
# that machine's Radeon Pro 560 -- a third-party AMD GPU -- was immature under PyTorch's MPS backend).
# Apple Silicon's own GPU is a different story. If this project ever moves to another machine again,
# re-benchmark before assuming either way -- don't carry this value over uncritically.
