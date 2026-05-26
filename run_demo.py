"""Run concave packing demo and generate visualizations."""

import os
import sys

src_dir = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_dir)

from uv_packing.config import PackingConfig, ModelSpec, Pose
from uv_packing.pose_sampler import sample_poses
from uv_packing.packer import concave_packer, save_results_json
from uv_packing.visualize import plot_packing_results, plot_packing_comparison

TEST_DATA = os.path.join(os.path.dirname(__file__), "tests", "test_data")

model_specs = {
    "l_shape": ModelSpec(
        model_id="l_shape",
        stl_path=os.path.join(TEST_DATA, "l_shape.stl"),
        poses=[
            Pose(0, 0, 0, 1, 0, 0, 0),
            Pose(0, 0, 5, 0.7071, 0, 0, 0.7071),
            Pose(0, 0, -5, 0.7071, 0, 0, -0.7071),
            Pose(0, 0, 10, 0, 0, 0, 1),
        ],
    ),
    "t_shape": ModelSpec(
        model_id="t_shape",
        stl_path=os.path.join(TEST_DATA, "t_shape.stl"),
        poses=[
            Pose(0, 0, 0, 1, 0, 0, 0),
            Pose(0, 0, 3, 0.7071, 0, 0, 0.7071),
            Pose(0, 0, -3, 0, 0, 0, 1),
        ],
    ),
    "rectangle": ModelSpec(
        model_id="rectangle",
        stl_path=os.path.join(TEST_DATA, "rectangle.stl"),
        poses=[
            Pose(0, 0, 0, 1, 0, 0, 0),
            Pose(0, 0, 2, 0.9239, 0, 0, 0.3827),
        ],
    ),
    "cylinder": ModelSpec(
        model_id="cylinder",
        stl_path=os.path.join(TEST_DATA, "cylinder.stl"),
        poses=[
            Pose(0, 0, 0, 1, 0, 0, 0),
            Pose(0, 0, 4, 0.7071, 0.7071, 0, 0),
        ],
    ),
    "small_square": ModelSpec(
        model_id="small_square",
        stl_path=os.path.join(TEST_DATA, "small_square.stl"),
        poses=[Pose(0, 0, 0, 1, 0, 0, 0)],
    ),
}

config = PackingConfig(
    area_x=(-50.0, 100.0),
    area_y=(-50.0, 100.0),
    margin=2.0,
    boundary_margin=2.0,
    random_seed=42,
)

# ---- Demo 1: Concave benefit - L-shapes + small squares ----
print("=" * 60)
print("Demo 1: Concave packing benefit (8 parts)")
print("=" * 60)
model_ids = ["l_shape", "l_shape", "l_shape",
             "small_square", "small_square", "small_square",
             "small_square", "small_square"]
sampled = sample_poses(model_ids, model_specs, random_seed=42)
results = concave_packer(model_specs, model_ids, config, sampled_poses=sampled)
placed = sum(1 for r in results if r.placed)
print(f"Placed: {placed}/{len(results)}")

output1 = os.path.join(os.path.dirname(__file__), "tests", "packing_result.png")
plot_packing_results(results, model_specs, config, sampled, save_path=output1)
json1 = os.path.join(os.path.dirname(__file__), "tests", "packing_result.json")
save_results_json(results, json1)
print(f"Saved poses to {json1}")
print()

# ---- Demo 2: Mixed shapes ----
print("=" * 60)
print("Demo 2: Mixed shapes (8 parts)")
print("=" * 60)
model_ids = ["l_shape", "l_shape", "t_shape", "t_shape",
             "rectangle", "rectangle", "cylinder", "small_square"]
sampled = sample_poses(model_ids, model_specs, random_seed=42)
results = concave_packer(model_specs, model_ids, config, sampled_poses=sampled)
placed = sum(1 for r in results if r.placed)
print(f"Placed: {placed}/{len(results)}")

output2 = os.path.join(os.path.dirname(__file__), "tests", "mixed_packing.png")
plot_packing_results(results, model_specs, config, sampled,
                     title="Mixed Shape Packing (8 parts)", save_path=output2)
json2 = os.path.join(os.path.dirname(__file__), "tests", "mixed_packing.json")
save_results_json(results, json2)
print(f"Saved poses to {json2}")
print()

# ---- Demo 3: Concave vs Convex comparison ----
print("=" * 60)
print("Demo 3: Concave vs Convex hull comparison (8 parts)")
print("=" * 60)
model_ids = ["l_shape", "l_shape", "l_shape",
             "small_square", "small_square", "small_square",
             "small_square", "small_square"]
sampled = sample_poses(model_ids, model_specs, random_seed=42)
results = concave_packer(model_specs, model_ids, config, sampled_poses=sampled)
placed = sum(1 for r in results if r.placed)
print(f"Placed: {placed}/{len(results)}")

output3 = os.path.join(os.path.dirname(__file__), "tests", "concave_vs_convex.png")
plot_packing_comparison(results, config, model_specs, save_path=output3)
print()

print("Done! Generated:")
print(f"  1. {output1}")
print(f"  2. {output2}")
print(f"  3. {output3}")
