"""UV Packing - Concave 3D model packing on a 2D plane.

Inspired by Blender's UV Packing concave mode (xatlas-based bitmap occupancy packing).
"""

from .config import PackingConfig, ModelSpec, Pose, PlacementResult
from .stl_loader import load_stl, get_mesh_triangles_2d
from .pose_sampler import sample_poses
from .projection import project_mesh_to_2d, extract_concave_outline
from .packer import concave_packer
from .visualize import plot_packing_results, plot_packing_comparison

__all__ = [
    "PackingConfig",
    "ModelSpec",
    "Pose",
    "PlacementResult",
    "load_stl",
    "get_mesh_triangles_2d",
    "sample_poses",
    "project_mesh_to_2d",
    "extract_concave_outline",
    "concave_packer",
    "plot_packing_results",
    "plot_packing_comparison",
]
