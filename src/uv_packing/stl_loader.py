"""STL file loading and 2D triangle extraction."""

import numpy as np
import trimesh


def load_stl(stl_path: str) -> trimesh.Trimesh:
    """Load an STL file and return a trimesh object."""
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        # If the file contains multiple geometries, merge them
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {stl_path}")
        mesh = trimesh.util.concatenate(meshes)
    return mesh


def get_mesh_triangles_2d(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract triangulated 2D vertices from a mesh's XY projection.

    Returns array of shape (N, 3, 2) where N is the number of triangles,
    each triangle has 3 vertices, each vertex has (x, y) coordinates.
    """
    vertices_3d = mesh.vertices
    faces = mesh.faces
    triangles_2d = np.zeros((len(faces), 3, 2))
    for i, face in enumerate(faces):
        for j, vidx in enumerate(face):
            triangles_2d[i, j, 0] = vertices_3d[vidx][0]  # x
            triangles_2d[i, j, 1] = vertices_3d[vidx][1]  # y
    return triangles_2d


def rotate_mesh_vertices(mesh: trimesh.Trimesh, qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Apply quaternion rotation to all mesh vertices and return rotated vertices.

    Uses the standard quaternion rotation formula: v' = q * v * q^-1
    where v is treated as a pure quaternion (0, x, y, z).
    """
    vertices = mesh.vertices.copy().astype(np.float64)
    # Normalize quaternion
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Compute rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)

    # macOS Accelerate BLAS triggers spurious FP warnings on large matmul batches.
    # The results are correct; suppress the noise.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return vertices @ R.T
