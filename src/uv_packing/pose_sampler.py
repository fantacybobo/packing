"""Pose sampling logic."""

import random
from .config import Pose


def sample_poses(model_ids: list,
                 model_specs: dict,
                 random_seed: int = None) -> list:
    """Sample poses from model specifications.

    Args:
        model_ids: List of model IDs to sample (can contain duplicates).
        model_specs: Dict mapping model_id -> ModelSpec with pose lists.
        random_seed: Seed for reproducible sampling.

    Returns:
        List of (model_id, Pose, pose_index) tuples.
    """
    rng = random.Random(random_seed)
    samples = []
    for mid in model_ids:
        spec = model_specs[mid]
        if not spec.poses:
            raise ValueError(f"Model {mid} has no poses to sample from")
        idx = rng.randint(0, len(spec.poses) - 1)
        samples.append((mid, spec.poses[idx], idx))
    return samples
