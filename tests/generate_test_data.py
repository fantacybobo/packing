"""Generate test STL models and pose data for testing the concave packer.

Two modes:
  --simple   : Original small test shapes (default, for unit tests)
  --realistic: High-poly realistic models (~20k tris, sizes 10mm-300mm)

Creates STL files with various shapes (L-shaped, T-shaped, rectangle, circle, etc.)
that exercise the concave packing algorithm.
"""

import os
import sys
import json
import math
import numpy as np
import trimesh
from shapely.geometry import Polygon


# ---- Helpers ----

def _subdivide_to_target(mesh, target_tris=20000):
    """Subdivide mesh until it reaches approximately target_tris triangles.

    Stops early if the next subdivision would overshoot target by more than 50%,
    keeping triangle counts in a reasonable range around the target.
    """
    while len(mesh.faces) < target_tris:
        next_count = len(mesh.faces) * 4
        # Avoid massive overshoot: stop if current is within 40% of target
        # and next step would more than double beyond target
        if next_count > target_tris * 2.0 and len(mesh.faces) >= target_tris * 0.4:
            break
        mesh = mesh.subdivide()
    return mesh


# ---- Simple test shapes (original) ----

def _create_box_mesh(width, depth, height, center=(0, 0, 0)):
    """Create a box trimesh."""
    mesh = trimesh.creation.box(extents=[width, depth, height])
    cx, cy, cz = center
    mesh.apply_translation([cx, cy, cz])
    return mesh


def _create_cylinder_mesh(radius, height, sections=32):
    """Create a cylinder trimesh."""
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def _create_l_shape(width, leg_w, height, center=(0, 0, 0)):
    """Create an L-shaped mesh (good for testing concave packing).

    The L-shape is in the XY plane extrusion, so its 2D projection is L-shaped.
    """
    outer_w = width

    # Outer polygon vertices (counter-clockwise)
    poly = Polygon([
        (0, 0),
        (outer_w, 0),
        (outer_w, leg_w),
        (leg_w, leg_w),
        (leg_w, outer_w),
        (0, outer_w),
    ])

    # Extrude in Z
    mesh = trimesh.creation.extrude_polygon(poly, height)
    cx, cy, _ = center
    mesh.apply_translation([cx - outer_w / 2, cy - outer_w / 2, -height / 2])
    return mesh


def _create_t_shape(stem_w, cross_w, stem_h, cross_h, height, center=(0, 0, 0)):
    """Create a T-shaped mesh."""
    total_w = cross_w
    total_h = stem_h + cross_h
    half_cross = cross_w / 2
    half_stem = stem_w / 2

    poly = Polygon([
        (half_cross - half_stem, 0),
        (half_cross + half_stem, 0),
        (half_cross + half_stem, cross_h),
        (cross_w, cross_h),
        (cross_w, cross_h + stem_h),
        (0, cross_h + stem_h),
        (0, cross_h),
        (half_cross - half_stem, cross_h),
    ])

    mesh = trimesh.creation.extrude_polygon(poly, height)
    cx, cy, _ = center
    mesh.apply_translation([cx - total_w / 2, cy - total_h / 2, -height / 2])
    return mesh


# ---- Realistic test shapes (~20k tris) ----

def _create_realistic_l_bracket(width, leg_w, height, name="l_bracket"):
    """Create a realistic L-shaped bracket with ~20k triangles.

    Args:
        width: Overall width/depth of the bracket (mm).
        leg_w: Width of each leg (mm).
        height: Extrusion height / thickness (mm).
    """
    poly = Polygon([
        (0, 0),
        (width, 0),
        (width, leg_w),
        (leg_w, leg_w),
        (leg_w, width),
        (0, width),
    ])
    mesh = trimesh.creation.extrude_polygon(poly, height)
    mesh = _subdivide_to_target(mesh, 20000)
    # Center the mesh
    cx, cy, cz = width / 2, width / 2, height / 2
    mesh.apply_translation([-cx, -cy, -cz])
    return mesh


def _create_realistic_t_bracket(stem_w, cross_w, stem_h, cross_h, height, name="t_bracket"):
    """Create a realistic T-shaped bracket with ~20k triangles."""
    total_w = cross_w
    total_h = stem_h + cross_h
    half_cross = cross_w / 2
    half_stem = stem_w / 2

    poly = Polygon([
        (half_cross - half_stem, 0),
        (half_cross + half_stem, 0),
        (half_cross + half_stem, cross_h),
        (cross_w, cross_h),
        (cross_w, cross_h + stem_h),
        (0, cross_h + stem_h),
        (0, cross_h),
        (half_cross - half_stem, cross_h),
    ])

    mesh = trimesh.creation.extrude_polygon(poly, height)
    mesh = _subdivide_to_target(mesh, 20000)
    cx, cy, cz = total_w / 2, total_h / 2, height / 2
    mesh.apply_translation([-cx, -cy, -cz])
    return mesh


def _create_realistic_flat_plate(width, depth, height, name="flat_plate"):
    """Create a realistic flat plate with ~20k triangles."""
    mesh = trimesh.creation.box(extents=[width, depth, height])
    mesh = _subdivide_to_target(mesh, 18000)
    return mesh


def _create_realistic_bowl(outer_radius, inner_radius, height, name="bowl"):
    """Create a realistic bowl/ring shape (annulus) with ~20k triangles.

    An annulus (thick ring) represents a bowl when projected from above.
    """
    mesh = trimesh.creation.annulus(
        r_min=inner_radius, r_max=outer_radius, height=height, sections=64
    )
    mesh = _subdivide_to_target(mesh, 20000)
    return mesh


def _create_realistic_cylinder(radius, height, name="cylinder"):
    """Create a realistic cylinder with ~20k triangles."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    mesh = _subdivide_to_target(mesh, 20000)
    return mesh


# ---- Pose helpers ----

def _make_pose(dx=0.0, dy=0.0, dz=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    return {"dx": dx, "dy": dy, "dz": dz, "qw": qw, "qx": qx, "qy": qy, "qz": qz}


def _rot_z_deg(deg, dx=0.0, dy=0.0, dz=0.0):
    """Quaternion rotation around Z axis by degrees."""
    half = math.radians(deg) / 2
    return _make_pose(dx=dx, dy=dy, dz=dz, qw=math.cos(half), qz=math.sin(half))


def _rot_x_deg(deg, dx=0.0, dy=0.0, dz=0.0):
    """Quaternion rotation around X axis by degrees (e.g., bowl on its side)."""
    half = math.radians(deg) / 2
    return _make_pose(dx=dx, dy=dy, dz=dz, qw=math.cos(half), qx=math.sin(half))


# ============================================================
# Simple test data (original, for unit tests)
# ============================================================

def generate_simple_test_data(output_dir: str):
    """Generate original simple test STL files and pose data."""
    os.makedirs(output_dir, exist_ok=True)

    test_models = {}
    config_data = {
        "models": {},
        "packing_area": {"x": [-300, 300], "y": [-200, 200]},
        "margin": 2.0,
        "test_scenarios": [],
    }

    # Model 1: L-shaped part
    l_shape = _create_l_shape(width=40.0, leg_w=15.0, height=10.0, center=(0, 0, 0))
    l_stl_path = os.path.join(output_dir, "l_shape.stl")
    l_shape.export(l_stl_path)

    l_poses = [
        _make_pose(),
        _make_pose(dz=5.0, qw=0.7071, qz=0.7071),
        _make_pose(dz=-5.0, qw=0.7071, qz=-0.7071),
        _make_pose(dz=10.0, qz=1.0),
    ]
    test_models["l_shape"] = {"stl_path": l_stl_path, "poses": l_poses}

    # Model 2: T-shaped part
    t_shape = _create_t_shape(stem_w=15.0, cross_w=50.0, stem_h=30.0, cross_h=15.0,
                               height=10.0, center=(0, 0, 0))
    t_stl_path = os.path.join(output_dir, "t_shape.stl")
    t_shape.export(t_stl_path)

    t_poses = [
        _make_pose(),
        _make_pose(dz=3.0, qw=0.7071, qz=0.7071),
        _make_pose(dz=-3.0, qz=1.0),
    ]
    test_models["t_shape"] = {"stl_path": t_stl_path, "poses": t_poses}

    # Model 3: Rectangle
    rect = _create_box_mesh(30.0, 20.0, 5.0, center=(0, 0, 0))
    rect_stl_path = os.path.join(output_dir, "rectangle.stl")
    rect.export(rect_stl_path)

    rect_poses = [
        _make_pose(),
        _make_pose(dz=2.0, qw=0.9239, qz=0.3827),
    ]
    test_models["rectangle"] = {"stl_path": rect_stl_path, "poses": rect_poses}

    # Model 4: Cylinder
    cyl = _create_cylinder_mesh(radius=15.0, height=8.0)
    cyl_stl_path = os.path.join(output_dir, "cylinder.stl")
    cyl.export(cyl_stl_path)

    cyl_poses = [
        _make_pose(),
        _make_pose(dz=4.0, qw=0.7071, qx=0.7071),
    ]
    test_models["cylinder"] = {"stl_path": cyl_stl_path, "poses": cyl_poses}

    # Model 5: Small square
    small_sq = _create_box_mesh(12.0, 12.0, 3.0, center=(0, 0, 0))
    small_stl_path = os.path.join(output_dir, "small_square.stl")
    small_sq.export(small_stl_path)

    small_poses = [_make_pose()]
    test_models["small_square"] = {"stl_path": small_stl_path, "poses": small_poses}

    # Test scenarios
    scenarios = [
        {
            "name": "simple_two_shapes",
            "model_ids": ["l_shape", "rectangle"],
            "description": "Two simple shapes - L and rectangle"
        },
        {
            "name": "concave_benefit",
            "model_ids": ["l_shape", "l_shape", "small_square", "small_square"],
            "description": "L-shapes create concave regions that small squares can fill"
        },
        {
            "name": "mixed_shapes",
            "model_ids": ["l_shape", "t_shape", "rectangle", "cylinder", "small_square"],
            "description": "Mix of different shapes"
        },
        {
            "name": "many_parts",
            "model_ids": ["l_shape", "l_shape", "t_shape", "t_shape",
                          "rectangle", "rectangle", "small_square", "small_square",
                          "small_square", "small_square"],
            "description": "10 parts - stress test"
        },
        {
            "name": "with_rotations",
            "model_ids": ["l_shape", "l_shape", "t_shape"],
            "description": "Shapes with different rotations"
        },
        {
            "name": "repeated_ids",
            "model_ids": ["l_shape", "l_shape", "l_shape", "l_shape"],
            "description": "Same model repeated - different pose samples"
        },
    ]

    config_data["test_scenarios"] = scenarios
    config_data["models"] = {
        "l_shape": {"stl": "l_shape.stl", "pose_count": len(l_poses)},
        "t_shape": {"stl": "t_shape.stl", "pose_count": len(t_poses)},
        "rectangle": {"stl": "rectangle.stl", "pose_count": len(rect_poses)},
        "cylinder": {"stl": "cylinder.stl", "pose_count": len(cyl_poses)},
        "small_square": {"stl": "small_square.stl", "pose_count": len(small_poses)},
    }

    config_path = os.path.join(output_dir, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"Generated {len(test_models)} simple test models in {output_dir}")
    print(f"Test scenarios: {len(scenarios)}")
    return test_models, config_data


# ============================================================
# Realistic test data (~20k tris/part, sizes 10mm-300mm)
# ============================================================

def generate_realistic_test_data(output_dir: str):
    """Generate realistic high-poly test STL files.

    Models:
      - l_bracket_large:   200×200mm L-bracket, leg=40mm, height=30mm,   ~20k tris
      - l_bracket_medium:  100×100mm L-bracket, leg=20mm, height=15mm,   ~20k tris
      - l_bracket_small:    40×40mm  L-bracket, leg=8mm,  height=6mm,    ~20k tris
      - flat_plate_large:  300×200mm plate,  5mm thick,                  ~18k tris
      - flat_plate_medium: 120×80mm  plate,  3mm thick,                  ~18k tris
      - flat_plate_small:   50×30mm  plate,  2mm thick,                  ~18k tris
      - bowl_ring_large:    outer R=150mm, inner R=90mm, H=40mm,         ~20k tris
      - bowl_ring_medium:   outer R=60mm,  inner R=35mm, H=25mm,         ~20k tris
      - t_bracket_medium:   120×100mm T-bracket, height=15mm,            ~20k tris
      - cylinder_small:     R=15mm, H=20mm,                              ~20k tris
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Generating realistic high-poly test models (~20k triangles each)")
    print("=" * 70)

    models = {}

    # ---- L-Brackets (concave packing test) ----

    # Large: 200mm × 200mm
    print("Creating l_bracket_large (200×200mm)...")
    lb_large = _create_realistic_l_bracket(width=200.0, leg_w=40.0, height=30.0)
    path = os.path.join(output_dir, "l_bracket_large.stl")
    lb_large.export(path)
    faces = len(lb_large.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["l_bracket_large"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "200×200×30",
        "type": "L-bracket",
        "poses": [
            _make_pose(),                                         # flat on XY plane
            _rot_z_deg(90, dz=10.0),                              # rotated 90°
            _rot_z_deg(-90, dz=-10.0),                            # rotated -90°
            _rot_z_deg(180, dz=20.0),                             # rotated 180°
            _make_pose(dz=5.0, qw=0.9239, qz=0.3827),            # 45° around Z
        ],
    }

    # Medium: 100mm × 100mm
    print("Creating l_bracket_medium (100×100mm)...")
    lb_med = _create_realistic_l_bracket(width=100.0, leg_w=20.0, height=15.0)
    path = os.path.join(output_dir, "l_bracket_medium.stl")
    lb_med.export(path)
    faces = len(lb_med.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["l_bracket_medium"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "100×100×15",
        "type": "L-bracket",
        "poses": [
            _make_pose(),
            _rot_z_deg(90, dz=3.0),
            _rot_z_deg(-90, dz=-3.0),
            _rot_z_deg(180, dz=8.0),
            _make_pose(dz=2.0, qw=0.9239, qz=0.3827),
        ],
    }

    # Small: 40mm × 40mm
    print("Creating l_bracket_small (40×40mm)...")
    lb_small = _create_realistic_l_bracket(width=40.0, leg_w=8.0, height=6.0)
    path = os.path.join(output_dir, "l_bracket_small.stl")
    lb_small.export(path)
    faces = len(lb_small.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["l_bracket_small"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "40×40×6",
        "type": "L-bracket",
        "poses": [
            _make_pose(),
            _rot_z_deg(90, dz=2.0),
            _rot_z_deg(-90, dz=-2.0),
            _rot_z_deg(180, dz=4.0),
        ],
    }

    # ---- Flat Plates ----

    # Large: 300mm × 200mm × 5mm
    print("Creating flat_plate_large (300×200×5mm)...")
    fp_large = _create_realistic_flat_plate(width=300.0, depth=200.0, height=5.0)
    path = os.path.join(output_dir, "flat_plate_large.stl")
    fp_large.export(path)
    faces = len(fp_large.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["flat_plate_large"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "300×200×5",
        "type": "flat_plate",
        "poses": [
            _make_pose(),
            _rot_z_deg(90, dz=1.0),
            _rot_z_deg(45, dz=2.0),
            _rot_z_deg(-45, dz=-2.0),
        ],
    }

    # Medium: 120mm × 80mm × 3mm
    print("Creating flat_plate_medium (120×80×3mm)...")
    fp_med = _create_realistic_flat_plate(width=120.0, depth=80.0, height=3.0)
    path = os.path.join(output_dir, "flat_plate_medium.stl")
    fp_med.export(path)
    faces = len(fp_med.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["flat_plate_medium"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "120×80×3",
        "type": "flat_plate",
        "poses": [
            _make_pose(),
            _rot_z_deg(90, dz=0.5),
            _rot_z_deg(30, dz=1.0),
            _rot_z_deg(-60, dz=-1.0),
        ],
    }

    # Small: 50mm × 30mm × 2mm
    print("Creating flat_plate_small (50×30×2mm)...")
    fp_small = _create_realistic_flat_plate(width=50.0, depth=30.0, height=2.0)
    path = os.path.join(output_dir, "flat_plate_small.stl")
    fp_small.export(path)
    faces = len(fp_small.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["flat_plate_small"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "50×30×2",
        "type": "flat_plate",
        "poses": [
            _make_pose(),
            _rot_z_deg(90, dz=0.5),
            _rot_z_deg(45, dz=1.0),
        ],
    }

    # ---- Bowl/Ring shapes (annulus = hollow cylinder, like a bowl rim) ----

    # Large: outer R=150mm, inner R=90mm, H=40mm
    print("Creating bowl_ring_large (R=150/90mm, H=40mm)...")
    bowl_large = _create_realistic_bowl(outer_radius=150.0, inner_radius=90.0, height=40.0)
    path = os.path.join(output_dir, "bowl_ring_large.stl")
    bowl_large.export(path)
    faces = len(bowl_large.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["bowl_ring_large"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "OD=300, ID=180, H=40",
        "type": "bowl_ring",
        "poses": [
            _make_pose(),                                          # upright (ring projection)
            _rot_z_deg(45, dz=5.0),                                # rotated
            _rot_z_deg(-45, dz=-5.0),
        ],
    }

    # Medium: outer R=60mm, inner R=35mm, H=25mm
    print("Creating bowl_ring_medium (R=60/35mm, H=25mm)...")
    bowl_med = _create_realistic_bowl(outer_radius=60.0, inner_radius=35.0, height=25.0)
    path = os.path.join(output_dir, "bowl_ring_medium.stl")
    bowl_med.export(path)
    faces = len(bowl_med.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["bowl_ring_medium"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "OD=120, ID=70, H=25",
        "type": "bowl_ring",
        "poses": [
            _make_pose(),
            _rot_z_deg(60, dz=3.0),
            _rot_z_deg(-30, dz=-3.0),
        ],
    }

    # ---- T-Bracket ----

    # Medium: 120mm × 100mm
    print("Creating t_bracket_medium (120×100mm)...")
    tb_med = _create_realistic_t_bracket(
        stem_w=20.0, cross_w=120.0, stem_h=60.0, cross_h=40.0, height=15.0
    )
    path = os.path.join(output_dir, "t_bracket_medium.stl")
    tb_med.export(path)
    faces = len(tb_med.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["t_bracket_medium"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "120×100×15",
        "type": "T-bracket",
        "poses": [
            _make_pose(),
            _rot_z_deg(90, dz=3.0),
            _rot_z_deg(-90, dz=-3.0),
            _rot_z_deg(180, dz=8.0),
        ],
    }

    # ---- Small Cylinder ----

    # Small: R=15mm, H=20mm
    print("Creating cylinder_small (R=15mm, H=20mm)...")
    cyl_small = _create_realistic_cylinder(radius=15.0, height=20.0)
    path = os.path.join(output_dir, "cylinder_small.stl")
    cyl_small.export(path)
    faces = len(cyl_small.faces)
    print(f"  -> {faces} triangles, saved to {path}")
    models["cylinder_small"] = {
        "stl_path": path,
        "faces": faces,
        "size_mm": "D=30, H=20",
        "type": "cylinder",
        "poses": [
            _make_pose(),
            _rot_z_deg(45, dz=3.0),
            _rot_z_deg(-45, dz=-3.0),
        ],
    }

    # ---- Build test scenarios ----

    scenarios = [
        {
            "name": "realistic_10_parts",
            "model_ids": [
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
            ],
            "description": "10 realistic parts (one of each type) - L, T, plate, bowl, cylinder"
        },
        {
            "name": "realistic_15_parts",
            "model_ids": [
                # Large parts (3)
                "l_bracket_large",
                "flat_plate_large",
                "bowl_ring_large",
                # Medium parts (7)
                "l_bracket_medium", "l_bracket_medium",
                "flat_plate_medium", "flat_plate_medium",
                "bowl_ring_medium",
                "t_bracket_medium", "t_bracket_medium",
                # Small parts (5)
                "l_bracket_small", "l_bracket_small",
                "flat_plate_small", "flat_plate_small",
                "cylinder_small",
            ],
            "description": "15 realistic parts - mix of large (3), medium (7), small (5)"
        },
        {
            "name": "realistic_20_parts",
            "model_ids": [
                # Large parts (4)
                "l_bracket_large", "l_bracket_large",
                "flat_plate_large",
                "bowl_ring_large",
                # Medium parts (10)
                "l_bracket_medium", "l_bracket_medium", "l_bracket_medium",
                "flat_plate_medium", "flat_plate_medium", "flat_plate_medium",
                "bowl_ring_medium", "bowl_ring_medium",
                "t_bracket_medium", "t_bracket_medium",
                # Small parts (6)
                "l_bracket_small", "l_bracket_small",
                "flat_plate_small", "flat_plate_small", "flat_plate_small",
                "cylinder_small",
            ],
            "description": "20 realistic parts - large (4), medium (10), small (6)"
        },
        {
            "name": "realistic_l_shape_heavy",
            "model_ids": [
                "l_bracket_large", "l_bracket_large",
                "l_bracket_medium", "l_bracket_medium", "l_bracket_medium",
                "l_bracket_small", "l_bracket_small", "l_bracket_small", "l_bracket_small",
                "flat_plate_medium", "flat_plate_small", "flat_plate_small",
                "cylinder_small", "cylinder_small",
            ],
            "description": "14 parts - L-bracket heavy (best for concave packing demo)"
        },
    ]

    # Generate JSON config
    config_data = {
        "models": {name: {
            "stl": os.path.basename(info["stl_path"]),
            "faces": info["faces"],
            "size_mm": info["size_mm"],
            "type": info["type"],
            "pose_count": len(info["poses"]),
        } for name, info in models.items()},
        "packing_area": {"x": [-600, 600], "y": [-400, 400]},
        "margin": 2.0,
        "test_scenarios": scenarios,
    }

    config_path = os.path.join(output_dir, "realistic_test_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print()
    print(f"Generated {len(models)} realistic models in {output_dir}")
    total_tris = sum(info["faces"] for info in models.values())
    print(f"Total triangles across all models: {total_tris:,}")
    print(f"Test scenarios: {len(scenarios)}")
    print(f"Config saved to {config_path}")

    # Print summary table
    print()
    print(f"{'Model':<25s} {'Type':<12s} {'Size':<25s} {'Tris':>8s}")
    print("-" * 72)
    for name, info in sorted(models.items()):
        print(f"{name:<25s} {info['type']:<12s} {info['size_mm']:<25s} {info['faces']:>8,d}")

    return models, config_data


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate test data for concave packer")
    parser.add_argument("--realistic", action="store_true",
                        help="Generate realistic high-poly test models (~20k tris each)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: tests/test_data)")
    args = parser.parse_args()

    test_dir = args.output or os.path.join(os.path.dirname(__file__), "test_data")
    os.makedirs(test_dir, exist_ok=True)

    if args.realistic:
        generate_realistic_test_data(test_dir)
    else:
        generate_simple_test_data(test_dir)
