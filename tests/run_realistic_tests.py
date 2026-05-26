"""Run concave packing with realistic high-poly test models.

Tests packing of 10-20 parts with ~20k triangles each,
part sizes ranging from 15mm to 300mm.
"""

import os
import sys
import json
import time
import math

src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src_dir)

from uv_packing.config import PackingConfig, ModelSpec, Pose
from uv_packing.pose_sampler import sample_poses
from uv_packing.packer import concave_packer
from uv_packing.visualize import plot_packing_results, plot_packing_comparison

TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def load_realistic_model_specs():
    """Load model specs from generated realistic test data."""
    config_path = os.path.join(TEST_DATA, "realistic_test_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}. Run 'python tests/generate_test_data.py --realistic' first."
        )

    with open(config_path) as f:
        config_data = json.load(f)

    specs = {}
    for model_id, info in config_data["models"].items():
        stl_path = os.path.join(TEST_DATA, info["stl"])
        if not os.path.exists(stl_path):
            print(f"WARNING: STL not found: {stl_path}, skipping {model_id}")
            continue

        # The poses are stored in the generate script, not JSON.
        # We reconstruct them here from the model metadata.
        specs[model_id] = ModelSpec(
            model_id=model_id,
            stl_path=stl_path,
            poses=_get_poses_for_model(model_id),
        )

    return specs, config_data


def _get_poses_for_model(model_id):
    """Return appropriate poses for each model type."""
    # L-brackets
    if model_id.startswith("l_bracket"):
        if "large" in model_id:
            return [
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 10, math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)),
                Pose(0, 0, -10, math.cos(math.pi/4), 0, 0, -math.sin(math.pi/4)),
                Pose(0, 0, 20, 0, 0, 0, 1),
                Pose(0, 0, 5, 0.9239, 0, 0, 0.3827),
            ]
        elif "medium" in model_id:
            return [
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 3, math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)),
                Pose(0, 0, -3, math.cos(math.pi/4), 0, 0, -math.sin(math.pi/4)),
                Pose(0, 0, 8, 0, 0, 0, 1),
                Pose(0, 0, 2, 0.9239, 0, 0, 0.3827),
            ]
        else:  # small
            return [
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 2, math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)),
                Pose(0, 0, -2, math.cos(math.pi/4), 0, 0, -math.sin(math.pi/4)),
                Pose(0, 0, 4, 0, 0, 0, 1),
            ]

    # Flat plates
    if model_id.startswith("flat_plate"):
        if "large" in model_id:
            return [
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 1, math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)),
                Pose(0, 0, 2, 0.9239, 0, 0, 0.3827),
                Pose(0, 0, -2, 0.9239, 0, 0, -0.3827),
            ]
        elif "medium" in model_id:
            return [
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 0.5, math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)),
                Pose(0, 0, 1, math.cos(math.pi/6), 0, 0, math.sin(math.pi/6)),
                Pose(0, 0, -1, math.cos(math.pi/6), 0, 0, -math.sin(math.pi/6)),
            ]
        else:  # small
            return [
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 0.5, math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)),
                Pose(0, 0, 1, 0.9239, 0, 0, 0.3827),
            ]

    # Bowl/ring shapes
    if model_id.startswith("bowl_ring"):
        return [
            Pose(0, 0, 0, 1, 0, 0, 0),
            Pose(0, 0, 5, 0.9239, 0, 0, 0.3827),
            Pose(0, 0, -5, 0.9239, 0, 0, -0.3827),
        ]

    # T-bracket
    if model_id.startswith("t_bracket"):
        return [
            Pose(0, 0, 0, 1, 0, 0, 0),
            Pose(0, 0, 3, math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)),
            Pose(0, 0, -3, math.cos(math.pi/4), 0, 0, -math.sin(math.pi/4)),
            Pose(0, 0, 8, 0, 0, 0, 1),
        ]

    # Cylinder
    if model_id.startswith("cylinder"):
        return [
            Pose(0, 0, 0, 1, 0, 0, 0),
            Pose(0, 0, 3, 0.9239, 0, 0, 0.3827),
            Pose(0, 0, -3, 0.9239, 0, 0, -0.3827),
        ]

    raise ValueError(f"Unknown model type: {model_id}")


def run_scenario(name, model_ids, specs, config, output_suffix=""):
    """Run a single packing scenario."""
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"  Parts: {len(model_ids)}")
    print(f"  Packing area: {config.area_width}mm x {config.area_height}mm")
    print(f"  Estimating total area...")

    # Rough area estimate
    total_raw_area = 0
    for mid in model_ids:
        from uv_packing.stl_loader import load_stl
        from uv_packing.projection import project_mesh_to_2d, extract_concave_outline
        mesh = load_stl(specs[mid].stl_path)
        tris = project_mesh_to_2d(mesh.vertices, mesh.faces)
        outline, _ = extract_concave_outline(tris, simplify_tolerance=0.1)
        total_raw_area += outline.area
    print(f"  Total raw area: {total_raw_area:,.0f} mm²")

    t0 = time.time()
    sampled = sample_poses(model_ids, specs, random_seed=config.random_seed)
    results = concave_packer(specs, model_ids, config, sampled_poses=sampled)
    elapsed = time.time() - t0

    placed = [r for r in results if r.placed]
    failed = [r for r in results if not r.placed]

    print(f"\n  Results: {len(placed)}/{len(results)} placed ({elapsed:.1f}s)")
    if failed:
        print(f"  Failed: {[r.model_id for r in failed]}")

    # Per-model-type summary
    from collections import Counter
    model_counts = Counter(r.model_id for r in placed)
    for mid, count in sorted(model_counts.items()):
        info = ""
        if mid in specs:
            spec = specs[mid]
            info = f" [tris: ~{len(spec.poses)} poses]"
        print(f"    {mid}: {count} placed{info}")

    return results, sampled, elapsed


def main():
    print("Loading realistic model specs...")
    specs, config_data = load_realistic_model_specs()
    print(f"Loaded {len(specs)} models:")
    for mid, spec in specs.items():
        stl = os.path.basename(spec.stl_path)
        size = os.path.getsize(spec.stl_path)
        print(f"  {mid}: {stl} ({size/1024:.0f} KB, {len(spec.poses)} poses)")

    # Packing area: generous 1200x800mm for parts up to 300mm
    config = PackingConfig(
        area_x=(-600.0, 600.0),
        area_y=(-400.0, 400.0),
        margin=3.0,
        random_seed=42,
        max_placement_attempts=3000,
    )

    # ---- Scenario 1: 10 parts (one of each) ----
    model_ids_10 = [
        "l_bracket_large",
        "l_bracket_medium",
        "l_bracket_small",
        "flat_plate_large",
        "flat_plate_medium",
        "flat_plate_small",
        "bowl_ring_large",
        "bowl_ring_medium",
        "t_bracket_medium",
        "cylinder_small",
    ]
    results_10, sampled_10, t_10 = run_scenario(
        "realistic_10_parts", model_ids_10, specs, config
    )
    plot_packing_results(
        results_10, specs, config, sampled_10,
        title="Realistic Packing — 10 Parts (12k-33k tris each)",
        save_path=os.path.join(OUTPUT_DIR, "realistic_10_parts.png"),
    )
    plot_packing_comparison(
        results_10, config, specs,
        save_path=os.path.join(OUTPUT_DIR, "realistic_10_comparison.png"),
    )

    # ---- Scenario 2: 15 parts (user's target range) ----
    model_ids_15 = [
        # Large (3)
        "l_bracket_large",
        "flat_plate_large",
        "bowl_ring_large",
        # Medium (7)
        "l_bracket_medium", "l_bracket_medium",
        "flat_plate_medium", "flat_plate_medium",
        "bowl_ring_medium",
        "t_bracket_medium", "t_bracket_medium",
        # Small (5)
        "l_bracket_small", "l_bracket_small",
        "flat_plate_small", "flat_plate_small",
        "cylinder_small",
    ]
    results_15, sampled_15, t_15 = run_scenario(
        "realistic_15_parts", model_ids_15, specs, config
    )
    plot_packing_results(
        results_15, specs, config, sampled_15,
        title="Realistic Packing — 15 Parts (target range)",
        save_path=os.path.join(OUTPUT_DIR, "realistic_15_parts.png"),
    )

    # ---- Scenario 3: 20 parts (upper bound) ----
    model_ids_20 = [
        # Large (4)
        "l_bracket_large", "l_bracket_large",
        "flat_plate_large",
        "bowl_ring_large",
        # Medium (10)
        "l_bracket_medium", "l_bracket_medium", "l_bracket_medium",
        "flat_plate_medium", "flat_plate_medium", "flat_plate_medium",
        "bowl_ring_medium", "bowl_ring_medium",
        "t_bracket_medium", "t_bracket_medium",
        # Small (6)
        "l_bracket_small", "l_bracket_small",
        "flat_plate_small", "flat_plate_small", "flat_plate_small",
        "cylinder_small",
    ]
    results_20, sampled_20, t_20 = run_scenario(
        "realistic_20_parts", model_ids_20, specs, config
    )
    plot_packing_results(
        results_20, specs, config, sampled_20,
        title="Realistic Packing — 20 Parts (upper bound)",
        save_path=os.path.join(OUTPUT_DIR, "realistic_20_parts.png"),
    )

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, results in [
        ("10 parts", results_10),
        ("15 parts", results_15),
        ("20 parts", results_20),
    ]:
        placed = sum(1 for r in results if r.placed)
        total = len(results)
        print(f"  {name}: {placed}/{total} placed")

    print(f"\nOutput images saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
