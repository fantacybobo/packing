"""3D mesh -> 2D projection and concave silhouette extraction.

The key insight from Blender's approach: for concave packing, we keep ALL
triangles of the shape (not just the convex hull), allowing other shapes
to fit into concave regions.
"""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union


def project_mesh_to_2d(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Project 3D mesh vertices and faces to 2D by dropping Z.

    Args:
        vertices: (N, 3) array of 3D vertex positions.
        faces: (M, 3) array of face indices.

    Returns:
        Array of shape (M, 3, 2) with (x, y) for each triangle vertex.
    """
    triangles_2d = np.zeros((len(faces), 3, 2))
    for i, face in enumerate(faces):
        for j, vidx in enumerate(face):
            triangles_2d[i, j, 0] = vertices[vidx][0]
            triangles_2d[i, j, 1] = vertices[vidx][1]
    return triangles_2d


def extract_concave_outline(triangles_2d: np.ndarray, simplify_tolerance: float = 0.05) -> tuple:
    """Extract the exact concave outline from 2D triangles.

    Uses Shapely to union all triangles, producing the exact concave shape.
    This is equivalent to Blender's approach of using ALL triangles (not convex hull).

    Args:
        triangles_2d: (M, 3, 2) array of triangle vertices.
        simplify_tolerance: Tolerance for simplifying the union outline.

    Returns:
        (outline_polygon, triangles_list) where:
        - outline_polygon is a Shapely Polygon/MultiPolygon representing the exact shape
        - triangles_list is list of (3, 2) arrays for each triangle
    """
    polygons = []
    tri_list = []
    for tri in triangles_2d:
        p = Polygon([(tri[0, 0], tri[0, 1]),
                      (tri[1, 0], tri[1, 1]),
                      (tri[2, 0], tri[2, 1])])
        if p.is_valid and p.area > 1e-12:
            polygons.append(p)
            tri_list.append(tri)

    if not polygons:
        raise ValueError("No valid triangles found in projection")

    outline = unary_union(polygons)
    if simplify_tolerance > 0 and not outline.is_empty:
        outline = outline.simplify(simplify_tolerance, preserve_topology=True)

    return outline, tri_list


def polygon_to_triangles(polygon: Polygon) -> np.ndarray:
    """Triangulate a concave Shapely polygon into 2D triangles.

    Uses ear-clipping triangulation via Shapely's delaunay triangulation
    on the polygon vertices, then filters triangles inside the polygon.

    Args:
        polygon: A Shapely Polygon.

    Returns:
        Array of shape (N, 3, 2) of triangle vertices.
    """
    if polygon.is_empty:
        return np.zeros((0, 3, 2))

    # Handle MultiPolygon
    if isinstance(polygon, MultiPolygon):
        all_tris = []
        for p in polygon.geoms:
            all_tris.append(polygon_to_triangles(p))
        if not all_tris:
            return np.zeros((0, 3, 2))
        return np.concatenate(all_tris, axis=0)

    # Get exterior coords and triangulate using ear-clipping
    coords = np.array(polygon.exterior.coords)[:-1]  # Remove duplicate last point
    if len(coords) < 3:
        return np.zeros((0, 3, 2))

    tris = _ear_clip_triangulate(coords)
    return tris


def _ear_clip_triangulate(vertices: np.ndarray) -> np.ndarray:
    """Simple ear-clipping triangulation for 2D polygons.

    Args:
        vertices: (N, 2) array of polygon vertices in CCW order.

    Returns:
        (M, 3, 2) array of triangles.
    """
    n = len(vertices)
    if n == 3:
        return vertices.reshape(1, 3, 2)

    verts = vertices.tolist()
    tris = []
    indices = list(range(n))

    # Check if polygon is clockwise, reverse if needed
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1] - vertices[j, 0] * vertices[i, 1]
    if area < 0:
        verts.reverse()
        indices = list(range(n))

    remaining = indices[:]
    safety = 0
    while len(remaining) > 2 and safety < n * 3:
        safety += 1
        found_ear = False
        for k in range(len(remaining)):
            i_idx = remaining[k]
            j_idx = remaining[(k + 1) % len(remaining)]
            m_idx = remaining[(k + 2) % len(remaining)]

            vi = np.array(verts[i_idx])
            vj = np.array(verts[j_idx])
            vm = np.array(verts[m_idx])

            # Check if this is an ear (interior angle < 180)
            edge1 = vj - vi
            edge2 = vm - vj
            cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]

            if cross <= 0:
                continue  # Not convex, skip

            # Check no other vertex is inside this triangle
            ear_tri = Polygon([vi, vj, vm])
            has_interior = False
            for r_idx in remaining:
                if r_idx in (i_idx, j_idx, m_idx):
                    continue
                pt = verts[r_idx]
                if ear_tri.contains(Point(pt[0], pt[1])):
                    has_interior = True
                    break

            if not has_interior:
                tris.append(np.array([vi, vj, vm]))
                remaining.pop((k + 1) % len(remaining))
                found_ear = True
                break

        if not found_ear:
            break

    # If ear-clipping failed, do a simple fan triangulation
    if not tris and len(remaining) >= 3:
        v0 = np.array(verts[remaining[0]])
        for k in range(1, len(remaining) - 1):
            v1 = np.array(verts[remaining[k]])
            v2 = np.array(verts[remaining[k + 1]])
            tris.append(np.array([v0, v1, v2]))

    return np.array(tris) if tris else vertices[:3].reshape(1, 3, 2)


def get_triangle_bounds(triangles: np.ndarray) -> tuple:
    """Get the bounding box of a set of 2D triangles.

    Returns (min_x, min_y, max_x, max_y).
    """
    if len(triangles) == 0:
        return (0, 0, 0, 0)
    all_pts = triangles.reshape(-1, 2)
    return (float(all_pts[:, 0].min()), float(all_pts[:, 1].min()),
            float(all_pts[:, 0].max()), float(all_pts[:, 1].max()))
