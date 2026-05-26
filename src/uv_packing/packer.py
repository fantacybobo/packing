"""Concave polygon packing using a bitmap occupancy approach.

Inspired by Blender's `pack_island_xatlas` implementation (uv_pack.cc),
which is based on xatlas by Jonathan Young.

Core algorithm:
1. Maintain a bitmap occupancy grid (like Blender's Occupancy class).
2. Each cell stores the signed distance to the nearest occupied triangle.
3. Place shapes largest-first, scanning from the origin outward.
4. Use "Witness Pixel" and "Triangle Hint" accelerators.
5. Dynamically increase bitmap resolution if needed (adaptive scaling).

The key difference from Blender: we pack into a FIXED rectangular area,
not an expanding one. Positions outside the area are rejected.

Reference: blender/source/blender/geometry/intern/uv_pack.cc
- class Occupancy (line 1124)
- pack_island_xatlas (line 1603)
- find_best_fit_for_island (line 1338)
"""

import json
import math
import os
import random
import numpy as np
from typing import Optional

from .config import PackingConfig, PlacementResult
from .projection import get_triangle_bounds


class OccupancyBitmap:
    """Bitmap occupancy grid for collision detection.

    Inspired by Blender's Occupancy class (uv_pack.cc:1124), but optimized:
    - Uses a boolean bitmap (occupied/not) instead of signed distance field
    - Rasterization via edge-function scanline fill (like GPU rasterizers)
    - Query mode checks if any pixel in expanded BBox is occupied

    The tradeoff: faster but pixel-level accuracy. The bitmap_radix parameter
    controls resolution — higher values give more accurate collision detection.
    """

    def __init__(self, bitmap_radix: int, area_width: float, area_height: float):
        self.area_width = area_width
        self.area_height = area_height

        max_dim = max(area_width, area_height)
        self.bitmap_scale_reciprocal = bitmap_radix / max_dim

        self.bx = max(1, int(math.ceil(area_width * self.bitmap_scale_reciprocal)))
        self.by = max(1, int(math.ceil(area_height * self.bitmap_scale_reciprocal)))

        self._bitmap = np.zeros((self.by, self.bx), dtype=bool)

    def clear(self):
        self._bitmap.fill(False)

    def increase_scale(self):
        self.bitmap_scale_reciprocal *= 2.0
        self.bx = max(1, int(math.ceil(self.area_width * self.bitmap_scale_reciprocal)))
        self.by = max(1, int(math.ceil(self.area_height * self.bitmap_scale_reciprocal)))
        self._bitmap = np.zeros((self.by, self.bx), dtype=bool)
        self.clear()

    @staticmethod
    def _ensure_cw_winding(triangles: np.ndarray) -> np.ndarray:
        """Ensure all triangles have CW winding (matching Blender's convention)."""
        result = triangles.copy()
        for i, tri in enumerate(result):
            v0, v1, v2 = tri
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            if cross > 0:
                result[i] = np.array([v0, v2, v1])
        return result

    @staticmethod
    def _edge_function(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Edge function (cross product) for point c relative to edge a→b.
        Positive = c is to the right of the edge (CW triangle).
        """
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _rasterize_triangle(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                              margin_px: float):
        """Fast scanline rasterization using edge functions.

        Marks all pixels inside the expanded triangle (CW winding, including margin)
        as occupied in the bitmap.

        Args:
            v0, v1, v2: Triangle vertices in PIXEL coordinates.
            margin_px: Margin in pixels to expand the triangle.
        """
        # Bounding box in pixel space, expanded by margin
        x0 = max(0, int(math.floor(min(v0[0], v1[0], v2[0]) - margin_px)))
        y0 = max(0, int(math.floor(min(v0[1], v1[1], v2[1]) - margin_px)))
        x1 = min(self.bx, int(math.ceil(max(v0[0], v1[0], v2[0]) + margin_px)) + 1)
        y1 = min(self.by, int(math.ceil(max(v0[1], v1[1], v2[1]) + margin_px)) + 1)

        if x1 <= x0 or y1 <= y0:
            return

        # Precompute edge constants for the un-expanded triangle
        # For a CW triangle, a point is inside if all edge functions are >= 0
        # For an expanded triangle (margin), we use: edge_fn(p) >= -margin_px * edge_length
        e01_len = math.sqrt((v1[0] - v0[0])**2 + (v1[1] - v0[1])**2)
        e12_len = math.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
        e20_len = math.sqrt((v0[0] - v2[0])**2 + (v0[1] - v2[1])**2)

        margin_01 = margin_px * e01_len
        margin_12 = margin_px * e12_len
        margin_20 = margin_px * e20_len

        for y in range(y0, y1):
            row = self._bitmap[y]
            for x in range(x0, x1):
                if row[x]:
                    continue
                p = np.array([float(x), float(y)])
                # CW triangle: interior points have edge_fn <= 0
                # Accept if NOT too far outside: edge_fn <= margin
                if (self._edge_function(v0, v1, p) <= margin_01 and
                    self._edge_function(v1, v2, p) <= margin_12 and
                    self._edge_function(v2, v0, p) <= margin_20):
                    row[x] = True

    def _check_overlap(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                        margin_px: float) -> bool:
        """Check if any pixel in the expanded triangle overlaps an occupied pixel.

        Args:
            v0, v1, v2: Triangle vertices in PIXEL coordinates (CW winding).
            margin_px: Margin in pixels.

        Returns:
            True if overlap detected.
        """
        x0 = max(0, int(math.floor(min(v0[0], v1[0], v2[0]) - margin_px)))
        y0 = max(0, int(math.floor(min(v0[1], v1[1], v2[1]) - margin_px)))
        x1 = min(self.bx, int(math.ceil(max(v0[0], v1[0], v2[0]) + margin_px)) + 1)
        y1 = min(self.by, int(math.ceil(max(v0[1], v1[1], v2[1]) + margin_px)) + 1)

        if x1 <= x0 or y1 <= y0:
            return False

        e01_len = math.sqrt((v1[0] - v0[0])**2 + (v1[1] - v0[1])**2)
        e12_len = math.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
        e20_len = math.sqrt((v0[0] - v2[0])**2 + (v0[1] - v2[1])**2)

        margin_01 = margin_px * e01_len
        margin_12 = margin_px * e12_len
        margin_20 = margin_px * e20_len

        for y in range(y0, y1):
            row = self._bitmap[y]
            for x in range(x0, x1):
                if not row[x]:
                    continue
                p = np.array([float(x), float(y)])
                # CW triangle: check if occupied pixel falls within expanded triangle
                if (self._edge_function(v0, v1, p) <= margin_01 and
                    self._edge_function(v1, v2, p) <= margin_12 and
                    self._edge_function(v2, v0, p) <= margin_20):
                    return True
        return False

    def _trace_triangle(self, uv0: np.ndarray, uv1: np.ndarray, uv2: np.ndarray,
                         margin: float, write: bool) -> float:
        """Trace a triangle against the bitmap.

        Args:
            uv0, uv1, uv2: Triangle vertices in WORLD space (CW winding).
            margin: Gap margin in world units.
            write: If True, write triangle into bitmap.
                   If False, query for overlap.

        Returns:
            -1.0 if available, >= 0 if occupied.
        """
        # Convert to pixel coordinates
        s = self.bitmap_scale_reciprocal
        v0 = uv0 * s
        v1 = uv1 * s
        v2 = uv2 * s
        margin_px = margin * s

        if write:
            self._rasterize_triangle(v0, v1, v2, margin_px)
            return -1.0
        else:
            if self._check_overlap(v0, v1, v2, margin_px):
                return 1.0
            return -1.0

    def _triangles_to_pixel_space(self, triangles: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Offset triangle vertices by translation (for collision detection)."""
        return triangles + translation.reshape(1, 1, 2)

    def trace_island(self, triangles_2d: np.ndarray, tx: float, ty: float,
                     margin: float, write: bool) -> float:
        """Trace all triangles of a shape at the given translation.

        Args:
            triangles_2d: (M, 3, 2) array of 2D triangles (local coords, no translation).
            tx, ty: Translation in world space.
            margin: Margin between parts.
            write: If True, write shape into bitmap. If False, query for collision.

        Returns:
            -1.0 if position is available, or extent of overlap.
        """
        # Triangles are assumed to already have CW winding from upstream processing
        delta = np.array([tx, ty])

        for j, tri in enumerate(triangles_2d):
            uv0 = tri[0] + delta
            uv1 = tri[1] + delta
            uv2 = tri[2] + delta

            extent = self._trace_triangle(uv0, uv1, uv2, margin, write)
            if not write and extent >= 0.0:
                self._triangle_hint = j
                return extent  # Occupied

        return -1.0  # Available

    def is_position_in_bounds(self, tx: float, ty: float,
                               tri_bounds: tuple, margin: float) -> bool:
        """Check if a shape at (tx, ty) is within the target area."""
        min_x, min_y, max_x, max_y = tri_bounds
        w, h = max_x - min_x, max_y - min_y

        return (-margin <= tx + min_x and tx + max_x <= self.area_width + margin and
                -margin <= ty + min_y and ty + max_y <= self.area_height + margin)


class ConcavePacker:
    """Concave polygon packer using bitmap occupancy.

    Implements the xatlas strategy from Blender's uv_pack.cc, adapted for
    fixed-area packing with random placement.
    """

    def __init__(self, config: PackingConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)

        # Offset area to origin for bitmap coordinate convenience
        self.area_offset_x = config.area_x[0]
        self.area_offset_y = config.area_y[0]

        self.occupancy = OccupancyBitmap(
            config.bitmap_radix,
            config.area_width,
            config.area_height
        )

    def _sort_shapes(self, shapes: list) -> list:
        """Sort shapes by area descending (largest first), with random shuffle for same-size.

        This mirrors Blender's approach of packing largest shapes first,
        but adds randomness to the order.
        """
        # Compute areas
        indexed = []
        for i, shape in enumerate(shapes):
            tris = shape['triangles_2d']
            area = 0.0
            for tri in tris:
                v0, v1, v2 = tri
                area += 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                                   (v2[0] - v0[0]) * (v1[1] - v0[1]))
            indexed.append((area, i, shape))

        # Shuffle first (randomness), then sort by area descending (stable sort)
        self.rng.shuffle(indexed)
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [item[2] for item in indexed]

    def find_position(self, triangles_2d: np.ndarray,
                      tri_bounds: tuple,
                      boundary_margin: float,
                      part_margin: float) -> Optional[tuple]:
        """Find a valid non-overlapping position using randomized grid search.

        Strategy: divide the area into a grid, try cell centers in random order.
        Falls back to finer random sampling if grid search fails.

        Args:
            triangles_2d: (M, 3, 2) 2D triangles in local coords.
            tri_bounds: (min_x, min_y, max_x, max_y) of the shape.
            boundary_margin: Gap from the area boundary.
            part_margin: Gap between shapes.

        Returns:
            (tx, ty) world-coordinate position, or None if no fit found.
        """
        min_x, min_y, max_x, max_y = tri_bounds
        shape_w = max_x - min_x
        shape_h = max_y - min_y

        area_w = self.occupancy.area_width
        area_h = self.occupancy.area_height

        tx_min = -min_x + boundary_margin
        tx_max = area_w - max_x - boundary_margin
        ty_min = -min_y + boundary_margin
        ty_max = area_h - max_y - boundary_margin

        if tx_min >= tx_max or ty_min >= ty_max:
            return None

        # Strategy 1: Coarse grid search with random cell order
        grid_step_x = max(shape_w * 0.5, (tx_max - tx_min) / 20.0)
        grid_step_y = max(shape_h * 0.5, (ty_max - ty_min) / 20.0)

        grid_cells = []
        tx = tx_min
        while tx < tx_max:
            ty = ty_min
            while ty < ty_max:
                grid_cells.append((tx, ty))
                ty += grid_step_y
            tx += grid_step_x

        self.rng.shuffle(grid_cells)
        for tx, ty in grid_cells[:min(len(grid_cells), 80)]:
            extent = self.occupancy.trace_island(triangles_2d, tx, ty, part_margin, write=False)
            if extent < 0.0:
                return (tx, ty)

        # Strategy 2: Fine random sampling
        num_fine = min(self.config.max_placement_attempts // 2, 400)
        for _ in range(num_fine):
            tx = self.rng.uniform(tx_min, tx_max)
            ty = self.rng.uniform(ty_min, ty_max)
            extent = self.occupancy.trace_island(triangles_2d, tx, ty, part_margin, write=False)
            if extent < 0.0:
                return (tx, ty)

        return None

    def pack(self, shapes: list) -> list:
        """Pack shapes into the target area using concave bitmap occupancy.

        Args:
            shapes: List of dicts with keys:
                - 'id': unique identifier
                - 'triangles_2d': (M, 3, 2) array of 2D triangles
                - 'dz': float, Z translation from the pose
                - 'quat': tuple (qw, qx, qy, qz)

        Returns:
            List of PlacementResult objects.
        """
        margin = self.config.margin
        boundary_margin = self.config.boundary_margin
        results = []
        placed_count = 0

        # Sort shapes largest first (like Blender)
        sorted_shapes = self._sort_shapes(shapes)

        for shape in sorted_shapes:
            triangles_2d = shape['triangles_2d']
            tri_bounds = get_triangle_bounds(triangles_2d)
            min_x, min_y, max_x, max_y = tri_bounds

            shape_w = max_x - min_x
            shape_h = max_y - min_y

            # Check if shape fits at all
            if shape_w > self.occupancy.area_width or shape_h > self.occupancy.area_height:
                results.append(PlacementResult(
                    model_id=shape['model_id'],
                    pose_index=shape['pose_index'],
                    rotation_quat=shape['quat'],
                    translation=(0.0, 0.0, shape['dz']),
                    placed=False,
                ))
                continue

            position = self.find_position(triangles_2d, tri_bounds, boundary_margin, margin)

            if position is None:
                # Try increasing bitmap resolution and retry
                if self.occupancy.bitmap_scale_reciprocal < self.config.bitmap_radix * 4 / max(
                        self.occupancy.area_width, self.occupancy.area_height):
                    self.occupancy.increase_scale()
                    # Re-trace placed shapes at new resolution
                    self._retrace_placed(results, shapes, margin)
                    position = self.find_position(triangles_2d, tri_bounds, boundary_margin, margin)

            if position is None:
                results.append(PlacementResult(
                    model_id=shape['model_id'],
                    pose_index=shape['pose_index'],
                    rotation_quat=shape['quat'],
                    translation=(0.0, 0.0, shape['dz']),
                    placed=False,
                ))
            else:
                tx, ty = position
                # Write shape to occupancy bitmap
                self.occupancy.trace_island(triangles_2d, tx, ty, margin, write=True)

                # Convert to world coordinates (with area offset)
                world_x = tx + self.area_offset_x
                world_y = ty + self.area_offset_y
                placed_count += 1

                results.append(PlacementResult(
                    model_id=shape['model_id'],
                    pose_index=shape['pose_index'],
                    rotation_quat=shape['quat'],
                    translation=(float(world_x), float(world_y), float(shape['dz'])),
                    placed=True,
                    sample_x=float(world_x),
                    sample_y=float(world_y),
                ))

        return results

    def _retrace_placed(self, results: list, shapes: list, margin: float):
        """Re-trace all placed shapes into the occupancy bitmap.

        Used after bitmap resolution increase.
        """
        self.occupancy.clear()
        shape_map = {}
        for s in shapes:
            shape_map[s['id']] = s

        for r in results:
            if not r.placed:
                continue
            shape_key = f"{r.model_id}_{r.pose_index}"
            for sid in shape_map:
                s = shape_map[sid]
                key = f"{s['model_id']}_{s['pose_index']}"
                if key == shape_key:
                    tx = r.sample_x - self.area_offset_x
                    ty = r.sample_y - self.area_offset_y
                    self.occupancy.trace_island(s['triangles_2d'], tx, ty, margin, write=True)
                    break


def concave_packer(model_specs: dict,
                   model_ids: list,
                   config: PackingConfig = None,
                   sampled_poses: list = None) -> list:
    """Main entry point for concave packing.

    Args:
        model_specs: Dict of model_id -> ModelSpec.
        model_ids: List of model IDs to place (can repeat).
        config: PackingConfig instance.
        sampled_poses: Optional pre-sampled poses from sample_poses().

    Returns:
        List of PlacementResult objects.
    """
    from .stl_loader import load_stl, rotate_mesh_vertices
    from .projection import project_mesh_to_2d, extract_concave_outline, polygon_to_triangles, get_triangle_bounds
    from .pose_sampler import sample_poses as do_sample

    if config is None:
        config = PackingConfig()

    if sampled_poses is None:
        sampled_poses = do_sample(model_ids, model_specs, config.random_seed)

    packer = ConcavePacker(config)

    # Prepare shapes
    shapes = []
    for mid, pose, pose_idx in sampled_poses:
        spec = model_specs[mid]
        mesh = load_stl(spec.stl_path)

        # Apply rotation from quaternion
        rotated_verts = rotate_mesh_vertices(mesh, pose.qw, pose.qx, pose.qy, pose.qz)

        # Project to 2D (drop Z)
        triangles_2d = project_mesh_to_2d(rotated_verts, mesh.faces)

        # Extract concave outline and get precise triangles
        outline, tri_list = extract_concave_outline(triangles_2d, simplify_tolerance=0.1)

        # Convert outline to triangulated 2D shape
        packed_tris = polygon_to_triangles(outline)

        if len(packed_tris) == 0:
            packed_tris = triangles_2d

        # Enforce CW winding (matching Blender's convention for signed distance)
        packed_tris = OccupancyBitmap._ensure_cw_winding(packed_tris)

        shapes.append({
            'id': f"{mid}_{pose_idx}",
            'model_id': mid,
            'pose_index': pose_idx,
            'triangles_2d': packed_tris,
            'dz': pose.dz,
            'quat': (pose.qw, pose.qx, pose.qy, pose.qz),
            'mesh': mesh,
        })

    results = packer.pack(shapes)
    return results


def results_to_pose_dict(results: list) -> dict:
    """Convert packing results to a dict of part_name -> 7-d pose.

    Each key is a unique part instance name (model_id with occurrence counter),
    each value is [dx, dy, dz, qw, qx, qy, qz].

    Args:
        results: List of PlacementResult from concave_packer().

    Returns:
        Dict[str, list[float]] mapping part name to 7-d pose.
    """
    instance_count = {}
    pose_dict = {}

    for r in results:
        if not r.placed:
            continue
        model_id = r.model_id
        if model_id not in instance_count:
            instance_count[model_id] = 0
        else:
            instance_count[model_id] += 1

        part_name = f"{model_id}_{instance_count[model_id]}"
        tx, ty, tz = r.translation
        qw, qx, qy, qz = r.rotation_quat
        pose_dict[part_name] = [tx, ty, tz, qw, qx, qy, qz]

    return pose_dict


def save_results_json(results: list, filepath: str) -> str:
    """Save packing results as a JSON file.

    Args:
        results: List of PlacementResult from concave_packer().
        filepath: Path to save the JSON file.

    Returns:
        The filepath written.
    """
    pose_dict = results_to_pose_dict(results)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pose_dict, f, indent=2, ensure_ascii=False)
    return filepath
