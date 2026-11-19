"""Shared device-resolution logic for SamSegmenter and ZeroShotDetector.

cuda > mps > cpu. mps (Apple Silicon's Metal GPU backend) was missing
entirely before -- on an Intel Mac that's a no-op (mps is never available
there), but it meant switching to Apple Silicon hardware silently kept
running everything on CPU, none of the new GPU going used, until this was
added."""

import torch


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
