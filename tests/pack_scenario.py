"""Pack and visualize 20 parts from a JSON scenario file.

Usage:
    uv run python tests/pack_scenario.py

Reads tests/test_data/scenario_20_parts.json, runs concave packing,
and writes the visualization to tests/test_data/scenario_20_parts.png.
"""

import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uv_packing.config import PackingConfig, ModelSpec, Pose
from uv_packing.packer import concave_packer, save_results_json
from uv_packing.visualize import plot_packing_results


def load_scenario(json_path):
    with open(json_path) as f:
        data = json.load(f)

    test_dir = os.path.dirname(json_path)

    model_specs = {}
    for mid, info in data["models"].items():
        model_specs[mid] = ModelSpec(
            model_id=mid,
            stl_path=os.path.join(test_dir, info["stl"]),
            poses=[],  # Not used — poses come from parts list
        )

    # Build sampled_poses list from the JSON parts array
    sampled_poses = []
    for i, item in enumerate(data["parts"]):
        p = item["pose"]
        pose = Pose(p["dx"], p["dy"], p["dz"], p["qw"], p["qx"], p["qy"], p["qz"])
        sampled_poses.append((item["model"], pose, i))

    config = PackingConfig(
        area_x=tuple(data["packing_area"]["x"]),
        area_y=tuple(data["packing_area"]["y"]),
        margin=data["margin"],
        boundary_margin=data.get("boundary_margin"),
        random_seed=data["random_seed"],
        max_placement_attempts=data.get("max_placement_attempts", 3000),
    )

    return model_specs, sampled_poses, config


def main():
    json_path = os.path.join(os.path.dirname(__file__), "test_data", "scenario_20_parts.json")

    print(f"Loading {json_path} ...")
    model_specs, sampled_poses, config = load_scenario(json_path)

    model_ids = [mid for mid, _, _ in sampled_poses]
    print(f"Parts: {len(model_ids)}")
    print(f"Area:  {config.area_width}mm x {config.area_height}mm")

    # Pack
    results = concave_packer(model_specs, model_ids, config, sampled_poses=sampled_poses)

    # Report
    placed = [r for r in results if r.placed]
    failed = [r for r in results if not r.placed]
    print(f"Placed: {len(placed)}/{len(results)}")
    for r in failed:
        print(f"  FAILED: {r.model_id}")

    # Visualize
    out = os.path.join(os.path.dirname(__file__), "test_data", "scenario_20_parts.png")
    plot_packing_results(results, model_specs, config, sampled_poses,
                         title="20 Realistic Parts — Concave Packing",
                         save_path=out)
    print(f"Saved {out}")

    # Save pose JSON
    json_out = os.path.join(os.path.dirname(__file__), "test_data", "scenario_20_parts_result.json")
    save_results_json(results, json_out)
    print(f"Saved {json_out}")


if __name__ == "__main__":
    main()
