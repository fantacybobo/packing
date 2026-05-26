"""Visualize concave packing results with matplotlib.

Renders the 2D packing layout showing each placed part's exact concave shape,
with color coding, labels, and the target area boundary.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
from typing import Optional

from .config import PackingConfig, PlacementResult
from .projection import polygon_to_triangles


def plot_packing_results(results: list,
                         model_specs: dict,
                         config: PackingConfig,
                         sampled_poses: list = None,
                         title: str = "Concave Packing Result",
                         figsize: tuple = (12, 8),
                         show_labels: bool = True,
                         save_path: Optional[str] = None):
    """Visualize the 2D packing layout.

    Args:
        results: List of PlacementResult from concave_packer().
        model_specs: Dict of model_id -> ModelSpec.
        config: PackingConfig used for packing.
        sampled_poses: Optional list of (model_id, Pose, idx) tuples.
        title: Plot title.
        figsize: Figure size in inches.
        show_labels: Whether to show model_id labels on each shape.
        save_path: If provided, save figure to this path.
    """
    from .stl_loader import load_stl, rotate_mesh_vertices
    from .projection import project_mesh_to_2d, extract_concave_outline

    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=figsize,
                                               gridspec_kw={'width_ratios': [3, 1]})

    # Draw target area boundary
    x0, x1 = config.area_x
    y0, y1 = config.area_y
    ax_main.add_patch(Rectangle(
        (x0, y0), config.area_width, config.area_height,
        fill=False, edgecolor='black', linewidth=2, linestyle='--',
        label=f'Target Area [{x0}:{x1}, {y0}:{y1}] mm'
    ))

    # Draw boundary margin inset
    bm = config.boundary_margin
    ax_main.add_patch(Rectangle(
        (x0 + bm, y0 + bm), config.area_width - 2 * bm, config.area_height - 2 * bm,
        fill=False, edgecolor='gray', linewidth=1, linestyle=':',
        label=f'Boundary margin: {bm} mm'
    ))

    # Color palette
    cmap = plt.cm.tab10
    color_map = {}
    color_idx = 0

    placed_count = 0
    failed_count = 0
    legend_handles = []

    # Build a mapping from result to its shape data
    pose_map = {}
    if sampled_poses:
        for mid, pose, idx in sampled_poses:
            key = (mid, idx)
            pose_map[key] = pose

    for r in results:
        # Determine color by model_id
        if r.model_id not in color_map:
            color_map[r.model_id] = cmap(color_idx % 10)
            color_idx += 1

        color = color_map[r.model_id]
        alpha = 0.6 if r.placed else 0.15

        if r.placed:
            # Reconstruct the 2D shape at its placed position
            spec = model_specs[r.model_id]
            mesh = load_stl(spec.stl_path)

            # Apply rotation
            q = r.rotation_quat
            rotated_verts = rotate_mesh_vertices(mesh, q[0], q[1], q[2], q[3])

            # Project to 2D
            tris_2d = project_mesh_to_2d(rotated_verts, mesh.faces)
            outline, _ = extract_concave_outline(tris_2d, simplify_tolerance=0.1)

            # Translate to placed position
            if outline.geom_type == 'Polygon':
                translated = _translate_polygon(outline, r.sample_x, r.sample_y)
                _draw_polygon(ax_main, translated, color, alpha, r.model_id, show_labels)
            elif outline.geom_type == 'MultiPolygon':
                for p in outline.geoms:
                    translated = _translate_polygon(p, r.sample_x, r.sample_y)
                    _draw_polygon(ax_main, translated, color, alpha, r.model_id, show_labels)

            placed_count += 1
        else:
            # Show failed items as small markers at origin
            ax_main.scatter([0], [0], marker='x', color='red', s=100, alpha=0.5)
            ax_main.annotate(f"FAIL: {r.model_id}", (0, 0 + failed_count * 10),
                           fontsize=8, color='red', alpha=0.7)
            failed_count += 1

    # Build legend
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis('off')
    ax_legend.set_title("Models", fontsize=12)

    y_pos = 0.95
    for model_id, color in color_map.items():
        ax_legend.add_patch(Rectangle(
            (0.05, y_pos - 0.04), 0.15, 0.06,
            facecolor=color, edgecolor='black', linewidth=0.5,
            transform=ax_legend.transAxes
        ))
        ax_legend.text(0.25, y_pos - 0.01, model_id, transform=ax_legend.transAxes,
                      fontsize=9, va='center')
        y_pos -= 0.08

    # Stats
    total = len(results)
    ax_legend.text(0.05, y_pos - 0.02, f"Placed: {placed_count}/{total}",
                  transform=ax_legend.transAxes, fontsize=10,
                  color='green' if failed_count == 0 else 'orange')
    if failed_count > 0:
        ax_legend.text(0.05, y_pos - 0.09, f"Failed: {failed_count}",
                      transform=ax_legend.transAxes, fontsize=10, color='red')

    # Also show Z distribution
    z_values = [r.translation[2] for r in results if r.placed]
    if z_values:
        ax_legend.text(0.05, y_pos - 0.18, f"Z range: [{min(z_values):.1f}, {max(z_values):.1f}] mm",
                      transform=ax_legend.transAxes, fontsize=9, color='gray')

    ax_main.set_xlim(x0 - 20, x1 + 20)
    ax_main.set_ylim(y0 - 20, y1 + 20)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X (mm)')
    ax_main.set_ylabel('Y (mm)')
    ax_main.set_title(title)
    ax_main.grid(True, alpha=0.3)

    fig.suptitle(f"Concave Packing — {placed_count}/{total} parts placed, "
                 f"area: {config.area_width}x{config.area_height} mm",
                 fontsize=10, y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    return fig


def plot_packing_comparison(concave_results: list,
                            config: PackingConfig,
                            model_specs: dict,
                            save_path: Optional[str] = None):
    """Compare concave packing against what convex hull packing would look like.

    Shows the exact shape (concave) as filled polygons and the convex hull
    as dashed outlines, demonstrating the space savings of concave mode.
    """
    from .stl_loader import load_stl, rotate_mesh_vertices
    from .projection import project_mesh_to_2d, extract_concave_outline

    fig, (ax_concave, ax_ch) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, mode in [(ax_concave, 'concave'), (ax_ch, 'convex')]:
        x0, x1 = config.area_x
        y0, y1 = config.area_y
        ax.add_patch(Rectangle(
            (x0, y0), config.area_width, config.area_height,
            fill=False, edgecolor='black', linewidth=2, linestyle='--'
        ))
        # Draw boundary margin inset
        bm = config.boundary_margin
        ax.add_patch(Rectangle(
            (x0 + bm, y0 + bm), config.area_width - 2 * bm, config.area_height - 2 * bm,
            fill=False, edgecolor='gray', linewidth=1, linestyle=':'
        ))
        ax.set_xlim(x0 - 20, x1 + 20)
        ax.set_ylim(y0 - 20, y1 + 20)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    cmap = plt.cm.tab10
    color_idx = 0

    for r in concave_results:
        if not r.placed:
            continue

        color = cmap(color_idx % 10)
        color_idx += 1

        spec = model_specs[r.model_id]
        mesh = load_stl(spec.stl_path)
        q = r.rotation_quat
        rotated_verts = rotate_mesh_vertices(mesh, q[0], q[1], q[2], q[3])
        tris_2d = project_mesh_to_2d(rotated_verts, mesh.faces)
        outline, _ = extract_concave_outline(tris_2d, simplify_tolerance=0.1)

        if outline.geom_type == 'Polygon':
            outline = outline
        elif outline.geom_type == 'MultiPolygon':
            outline = outline.geoms[0] if outline.geoms else outline

        # Draw concave shape
        translated = _translate_polygon(outline, r.sample_x, r.sample_y)
        _draw_polygon(ax_concave, translated, color, 0.6, r.model_id, True)

        # Draw convex hull
        ch = outline.convex_hull
        ch_translated = _translate_polygon(ch, r.sample_x, r.sample_y)
        _draw_polygon(ax_ch, ch_translated, color, 0.6, r.model_id, True)

        # Show the "wasted space" — difference between convex and concave
        if hasattr(ch, 'difference'):
            wasted = ch.difference(outline)
            if not wasted.is_empty:
                wasted_trans = _translate_polygon(wasted, r.sample_x, r.sample_y) if isinstance(wasted, Polygon) else wasted
                if isinstance(wasted_trans, Polygon):
                    ax_ch.add_patch(MplPolygon(
                        list(wasted_trans.exterior.coords),
                        facecolor='red', edgecolor='none', alpha=0.3, hatch='///'
                    ))

    ax_concave.set_title("Concave (Exact Shape) Packing")
    ax_ch.set_title("Convex Hull Packing\n(Red = wasted space)")

    total_convex_area = 0
    total_concave_area = 0
    for r in concave_results:
        if r.placed:
            spec = model_specs[r.model_id]
            mesh = load_stl(spec.stl_path)
            q = r.rotation_quat
            rotated_verts = rotate_mesh_vertices(mesh, q[0], q[1], q[2], q[3])
            tris_2d = project_mesh_to_2d(rotated_verts, mesh.faces)
            outline, _ = extract_concave_outline(tris_2d)
            total_concave_area += outline.area
            total_convex_area += outline.convex_hull.area

    waste_pct = (total_convex_area - total_concave_area) / total_convex_area * 100 if total_convex_area > 0 else 0
    fig.suptitle(f"Concave vs Convex Packing — "
                 f"Wasted space with convex: {waste_pct:.1f}%",
                 fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    return fig


def _translate_polygon(polygon: Polygon, dx: float, dy: float):
    """Translate a Shapely polygon."""
    from shapely import affinity
    return affinity.translate(polygon, xoff=dx, yoff=dy)


def _draw_polygon(ax, polygon: Polygon, color, alpha: float,
                  label: str, show_label: bool):
    """Draw a Shapely polygon on a matplotlib axis."""
    if polygon.is_empty:
        return

    coords = list(polygon.exterior.coords)
    patch = MplPolygon(coords, facecolor=color, edgecolor='black',
                       linewidth=0.8, alpha=alpha)
    ax.add_patch(patch)

    # Draw interior holes
    for interior in polygon.interiors:
        hole_coords = list(interior.coords)
        hole = MplPolygon(hole_coords, facecolor='white', edgecolor='black',
                         linewidth=0.5, alpha=alpha)
        ax.add_patch(hole)

    if show_label:
        centroid = polygon.centroid
        ax.annotate(label, (centroid.x, centroid.y),
                   fontsize=7, ha='center', va='center',
                   color='black', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
