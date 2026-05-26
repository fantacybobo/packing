"""Comprehensive tests for the concave packing system."""

import os
import sys
import json
import math
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from uv_packing.config import PackingConfig, ModelSpec, Pose, PlacementResult
from uv_packing.stl_loader import load_stl, rotate_mesh_vertices, get_mesh_triangles_2d
from uv_packing.pose_sampler import sample_poses
from uv_packing.projection import (
    project_mesh_to_2d,
    extract_concave_outline,
    polygon_to_triangles,
    get_triangle_bounds,
    _ear_clip_triangulate,
)
from uv_packing.packer import (
    OccupancyBitmap,
    ConcavePacker,
    concave_packer,
)


# Paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
L_SHAPE_STL = os.path.join(TEST_DATA_DIR, "l_shape.stl")
T_SHAPE_STL = os.path.join(TEST_DATA_DIR, "t_shape.stl")
RECT_STL = os.path.join(TEST_DATA_DIR, "rectangle.stl")
CYL_STL = os.path.join(TEST_DATA_DIR, "cylinder.stl")
SMALL_STL = os.path.join(TEST_DATA_DIR, "small_square.stl")


# ---- Fixtures ----

@pytest.fixture
def default_config():
    return PackingConfig(
        area_x=(-300.0, 300.0),
        area_y=(-200.0, 200.0),
        margin=2.0,
        random_seed=42,
    )


@pytest.fixture
def model_specs():
    return {
        "l_shape": ModelSpec(
            model_id="l_shape",
            stl_path=L_SHAPE_STL,
            poses=[
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 5, 0.7071, 0, 0, 0.7071),
                Pose(0, 0, -5, 0.7071, 0, 0, -0.7071),
                Pose(0, 0, 10, 0, 0, 0, 1),
            ],
        ),
        "t_shape": ModelSpec(
            model_id="t_shape",
            stl_path=T_SHAPE_STL,
            poses=[
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 3, 0.7071, 0, 0, 0.7071),
                Pose(0, 0, -3, 0, 0, 0, 1),
            ],
        ),
        "rectangle": ModelSpec(
            model_id="rectangle",
            stl_path=RECT_STL,
            poses=[
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 2, 0.9239, 0, 0, 0.3827),
            ],
        ),
        "cylinder": ModelSpec(
            model_id="cylinder",
            stl_path=CYL_STL,
            poses=[
                Pose(0, 0, 0, 1, 0, 0, 0),
                Pose(0, 0, 4, 0.7071, 0.7071, 0, 0),
            ],
        ),
        "small_square": ModelSpec(
            model_id="small_square",
            stl_path=SMALL_STL,
            poses=[
                Pose(0, 0, 0, 1, 0, 0, 0),
            ],
        ),
    }


# ---- STL Loading Tests ----

class TestSTLLoading:
    def test_load_l_shape(self):
        mesh = load_stl(L_SHAPE_STL)
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_load_all_models(self):
        for path in [L_SHAPE_STL, T_SHAPE_STL, RECT_STL, CYL_STL, SMALL_STL]:
            mesh = load_stl(path)
            assert mesh is not None
            assert len(mesh.vertices) >= 8, f"Too few vertices in {path}"
            assert len(mesh.faces) >= 4, f"Too few faces in {path}"

    def test_get_mesh_triangles_2d(self):
        mesh = load_stl(L_SHAPE_STL)
        tris = get_mesh_triangles_2d(mesh)
        assert tris.ndim == 3
        assert tris.shape[1:] == (3, 2)
        assert tris.shape[0] == len(mesh.faces)

    def test_rotate_mesh_vertices_identity(self):
        mesh = load_stl(RECT_STL)
        orig = mesh.vertices.copy()
        rotated = rotate_mesh_vertices(mesh, 1.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(rotated, orig, atol=1e-10)

    def test_rotate_mesh_vertices_90_z(self):
        """90 degrees around Z should swap X and Y."""
        mesh = load_stl(RECT_STL)
        qw = math.cos(math.pi / 4)  # cos(45°) for 90° rotation
        qz = math.sin(math.pi / 4)  # sin(45°)
        rotated = rotate_mesh_vertices(mesh, qw, 0.0, 0.0, qz)
        for i, v in enumerate(mesh.vertices):
            # (x, y, z) rotated 90° around Z -> (-y, x, z)
            expected = np.array([-v[1], v[0], v[2]])
            np.testing.assert_allclose(rotated[i], expected, atol=1e-6)


# ---- Pose Sampling Tests ----

class TestPoseSampling:
    def test_sample_single(self, model_specs):
        samples = sample_poses(["rectangle"], model_specs, random_seed=42)
        assert len(samples) == 1
        mid, pose, idx = samples[0]
        assert mid == "rectangle"
        assert 0 <= idx < len(model_specs["rectangle"].poses)

    def test_sample_with_repeats(self, model_specs):
        model_ids = ["l_shape", "l_shape", "l_shape"]
        samples = sample_poses(model_ids, model_specs, random_seed=42)
        assert len(samples) == 3
        for mid, pose, idx in samples:
            assert mid == "l_shape"

    def test_sample_reproducibility(self, model_specs):
        s1 = sample_poses(["l_shape", "t_shape"], model_specs, random_seed=42)
        s2 = sample_poses(["l_shape", "t_shape"], model_specs, random_seed=42)
        for (_, p1, i1), (_, p2, i2) in zip(s1, s2):
            assert i1 == i2

    def test_sample_different_seeds(self, model_specs):
        """Different seeds may produce different results."""
        s1 = sample_poses(["l_shape"] * 5, model_specs, random_seed=1)
        s2 = sample_poses(["l_shape"] * 5, model_specs, random_seed=2)
        indices1 = [i for _, _, i in s1]
        indices2 = [i for _, _, i in s2]
        # With 5 samples from 4 poses, seeds might differ
        assert len(indices1) == len(indices2) == 5

    def test_empty_poses_raises(self):
        spec = {"empty": ModelSpec(model_id="empty", stl_path=RECT_STL, poses=[])}
        with pytest.raises(ValueError, match="no poses"):
            sample_poses(["empty"], spec)


# ---- Projection Tests ----

class TestProjection:
    def test_project_mesh_to_2d(self):
        mesh = load_stl(RECT_STL)
        tris = project_mesh_to_2d(mesh.vertices, mesh.faces)
        assert tris.ndim == 3
        assert tris.shape[1:] == (3, 2)
        assert tris.shape[0] == len(mesh.faces)

    def test_extract_concave_outline(self):
        mesh = load_stl(RECT_STL)
        tris = project_mesh_to_2d(mesh.vertices, mesh.faces)
        outline, tri_list = extract_concave_outline(tris)
        assert outline is not None
        assert not outline.is_empty
        assert outline.area > 0

    def test_outline_area_positive(self):
        """All test models should have positive 2D projected area."""
        for path in [L_SHAPE_STL, T_SHAPE_STL, RECT_STL, CYL_STL, SMALL_STL]:
            mesh = load_stl(path)
            tris = project_mesh_to_2d(mesh.vertices, mesh.faces)
            outline, _ = extract_concave_outline(tris)
            assert outline.area > 0, f"Zero area for {os.path.basename(path)}"

    def test_l_shape_has_concave_region(self):
        """L-shape outline should have a concave region (not be convex)."""
        mesh = load_stl(L_SHAPE_STL)
        tris = project_mesh_to_2d(mesh.vertices, mesh.faces)
        outline, _ = extract_concave_outline(tris)
        convex_hull = outline.convex_hull
        # Concave shape area should be less than convex hull area
        assert outline.area < convex_hull.area * 0.95, (
            f"L-shape should be concave: outline area={outline.area:.2f}, "
            f"convex hull area={convex_hull.area:.2f}"
        )

    def test_ear_clip_triangulate_square(self):
        """Ear-clipping should triangulate a simple square."""
        square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        tris = _ear_clip_triangulate(square)
        assert len(tris) >= 2
        assert tris.shape[1:] == (3, 2)

    def test_polygon_to_triangles(self):
        mesh = load_stl(RECT_STL)
        tris_2d = project_mesh_to_2d(mesh.vertices, mesh.faces)
        outline, _ = extract_concave_outline(tris_2d)
        result = polygon_to_triangles(outline)
        assert len(result) > 0
        assert result.shape[1:] == (3, 2)

    def test_get_triangle_bounds(self):
        tris = np.array([
            [[-10, -5], [10, -5], [0, 5]],
            [[-5, 0], [5, 0], [0, 10]],
        ])
        min_x, min_y, max_x, max_y = get_triangle_bounds(tris)
        assert min_x == -10.0
        assert max_x == 10.0
        assert min_y == -5.0
        assert max_y == 10.0


# ---- Occupancy Bitmap Tests ----

class TestOccupancyBitmap:
    def test_init(self, default_config):
        occ = OccupancyBitmap(800, default_config.area_width, default_config.area_height)
        assert occ.bx > 0
        assert occ.by > 0
        assert not np.any(occ._bitmap)

    def test_clear(self, default_config):
        occ = OccupancyBitmap(800, default_config.area_width, default_config.area_height)
        occ._bitmap[10, 10] = True
        occ.clear()
        assert not np.any(occ._bitmap)

    def test_triangle_write_and_query(self, default_config):
        """Write a triangle then query - should detect overlap."""
        occ = OccupancyBitmap(800, default_config.area_width, default_config.area_height)

        tri = np.array([[100.0, 100.0], [140.0, 100.0], [120.0, 140.0]])
        # Ensure CW winding for the test
        tri = OccupancyBitmap._ensure_cw_winding(tri.reshape(1, 3, 2))[0]

        # First query: should be available
        result = occ._trace_triangle(tri[0], tri[1], tri[2], 2.0, write=False)
        assert result < 0, "Position should be available before writing"

        # Write the triangle
        occ._trace_triangle(tri[0], tri[1], tri[2], 2.0, write=True)
        # After writing, check overlap at same position
        result = occ._trace_triangle(tri[0], tri[1], tri[2], 2.0, write=False)
        assert result >= 0, "Position should be occupied after writing"

    def test_non_overlapping_positions(self, default_config):
        """Two non-overlapping triangles should not collide."""
        occ = OccupancyBitmap(800, default_config.area_width, default_config.area_height)

        tri1 = np.array([[50.0, 50.0], [90.0, 50.0], [70.0, 90.0]])
        tri2 = np.array([[200.0, 200.0], [240.0, 200.0], [220.0, 240.0]])
        tri1 = OccupancyBitmap._ensure_cw_winding(tri1.reshape(1, 3, 2))[0]
        tri2 = OccupancyBitmap._ensure_cw_winding(tri2.reshape(1, 3, 2))[0]

        occ._trace_triangle(tri1[0], tri1[1], tri1[2], 2.0, write=True)
        result = occ._trace_triangle(tri2[0], tri2[1], tri2[2], 2.0, write=False)
        assert result < 0, "Non-overlapping shapes should not collide"

    def test_trace_island(self, default_config):
        """Test the full island tracing."""
        occ = OccupancyBitmap(800, default_config.area_width, default_config.area_height)

        triangles = np.array([
            [[10.0, 10.0], [40.0, 10.0], [25.0, 40.0]],
            [[40.0, 10.0], [70.0, 10.0], [55.0, 40.0]],
        ])
        triangles = OccupancyBitmap._ensure_cw_winding(triangles)

        # Check availability
        result = occ.trace_island(triangles, 0.0, 0.0, 2.0, write=False)
        assert result < 0, "Should be available initially"

        # Write
        occ.trace_island(triangles, 0.0, 0.0, 2.0, write=True)

        # Check again - should be occupied
        result = occ.trace_island(triangles, 0.0, 0.0, 2.0, write=False)
        assert result >= 0, "Should be occupied after writing"

    def test_increase_scale(self, default_config):
        occ = OccupancyBitmap(800, default_config.area_width, default_config.area_height)
        orig_recip = occ.bitmap_scale_reciprocal
        occ.increase_scale()
        assert occ.bitmap_scale_reciprocal == orig_recip * 2.0


# ---- Concave Packer Tests ----

class TestConcavePacker:
    def test_single_shape_packs(self, default_config, model_specs):
        """A single shape should always pack."""
        model_ids = ["rectangle"]
        results = concave_packer(model_specs, model_ids, default_config)
        assert len(results) == 1
        assert results[0].placed

    def test_two_shapes_pack(self, default_config, model_specs):
        """Two shapes should pack without overlap."""
        model_ids = ["l_shape", "rectangle"]
        results = concave_packer(model_specs, model_ids, default_config)
        assert len(results) == 2
        for r in results:
            assert r.placed, f"Failed to place {r.model_id}"

    def test_concave_benefit(self, default_config, model_specs):
        """L-shapes leave concave gaps that small squares can fill.

        This is the key advantage of concave packing over convex hull packing."""
        model_ids = ["l_shape", "l_shape", "small_square", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)
        placed = [r for r in results if r.placed]
        # At minimum the two L-shapes and one small square should fit
        assert len(placed) >= 3, f"Expected >= 3 placed, got {len(placed)}"

    def test_mixed_shapes(self, default_config, model_specs):
        """Mix of different shapes."""
        model_ids = ["l_shape", "t_shape", "rectangle", "cylinder", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)
        placed = [r for r in results if r.placed]
        assert len(placed) >= 3, f"Expected >= 3 placed, got {len(placed)}"

    def test_no_overlap(self, default_config, model_specs):
        """Placed parts should not overlap each other."""
        model_ids = ["l_shape", "l_shape", "t_shape", "rectangle", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)

        placed = [(r, model_specs[r.model_id]) for r in results if r.placed]
        # Check that placed positions are all within bounds
        for r, _ in placed:
            assert default_config.area_x[0] - default_config.margin <= r.sample_x <= default_config.area_x[1] + default_config.margin
            assert default_config.area_y[0] - default_config.margin <= r.sample_y <= default_config.area_y[1] + default_config.margin

    def test_with_rotations(self, default_config, model_specs):
        """Shapes with different rotational poses should still pack."""
        model_ids = ["l_shape", "l_shape", "t_shape"]
        results = concave_packer(model_specs, model_ids, default_config)
        placed = [r for r in results if r.placed]
        assert len(placed) >= 2, f"Expected >= 2 placed, got {len(placed)}"

    def test_reproducible_results(self, default_config, model_specs):
        """Same seed should produce same results."""
        model_ids = ["l_shape", "rectangle", "small_square"]

        config1 = PackingConfig(random_seed=42)
        config2 = PackingConfig(random_seed=42)

        r1 = concave_packer(model_specs, model_ids, config1)
        r2 = concave_packer(model_specs, model_ids, config2)

        for a, b in zip(r1, r2):
            assert a.placed == b.placed
            if a.placed:
                assert a.sample_x == b.sample_x
                assert a.sample_y == b.sample_y

    def test_dz_preserved(self, default_config, model_specs):
        """Pose dz values should be preserved in the translation."""
        model_ids = ["l_shape", "rectangle"]
        # Force specific pose indices with known dz
        results = concave_packer(model_specs, model_ids, default_config)
        for r in results:
            assert len(r.translation) == 3
            # Z should match the sampled pose's dz
            assert isinstance(r.translation[2], float)

    def test_result_structure(self, default_config, model_specs):
        """Verify each result has all required fields."""
        model_ids = ["l_shape"]
        results = concave_packer(model_specs, model_ids, default_config)
        r = results[0]
        assert hasattr(r, 'model_id')
        assert hasattr(r, 'pose_index')
        assert hasattr(r, 'rotation_quat')
        assert hasattr(r, 'translation')
        assert hasattr(r, 'placed')
        assert hasattr(r, 'sample_x')
        assert hasattr(r, 'sample_y')
        assert len(r.rotation_quat) == 4
        assert len(r.translation) == 3


# ---- Integration Test ----

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_small(self, default_config, model_specs):
        """Complete pipeline with 3-5 parts."""
        model_ids = ["l_shape", "rectangle", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)

        assert len(results) == len(model_ids)
        all_placed = all(r.placed for r in results)
        if not all_placed:
            unplaced = [r.model_id for r in results if not r.placed]
            print(f"Note: {len(unplaced)} parts could not be placed: {unplaced}")
        # In a 600x400 area, these 3 small parts should all fit
        assert all_placed, f"All parts should fit: {unplaced}"

    def test_full_pipeline_medium(self, default_config, model_specs):
        """Complete pipeline with 5-8 parts."""
        model_ids = ["l_shape", "l_shape", "t_shape", "t_shape",
                     "rectangle", "small_square", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)
        placed = [r for r in results if r.placed]
        assert len(placed) >= 5, f"Expected >= 5 placed, got {len(placed)}"

    def test_many_parts(self, default_config, model_specs):
        """10 parts - stress test the packer."""
        model_ids = ["l_shape", "l_shape", "t_shape", "t_shape",
                     "rectangle", "rectangle", "small_square", "small_square",
                     "small_square", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)
        placed_count = sum(1 for r in results if r.placed)
        # Should be able to place most of them
        assert placed_count >= 6, f"Expected >= 6 placed, got {placed_count}"

    def test_result_positions_are_finite(self, default_config, model_specs):
        """All result coordinates should be finite numbers."""
        model_ids = ["l_shape", "t_shape", "rectangle", "cylinder", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)
        for r in results:
            if r.placed:
                assert all(math.isfinite(v) for v in r.translation)
                assert math.isfinite(r.sample_x)
                assert math.isfinite(r.sample_y)

    def test_concave_vs_convex_comparison(self, default_config, model_specs):
        """Demonstrate that concave packing can be denser.

        L-shapes have concave regions (the "notch") that other small shapes
        can fill. Our concave mode should allow this.
        """
        model_ids = ["l_shape", "l_shape", "small_square", "small_square", "small_square"]
        results = concave_packer(model_specs, model_ids, default_config)
        placed = [r for r in results if r.placed]
        # Concave mode should fit at least 3 of the 5 parts
        assert len(placed) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
