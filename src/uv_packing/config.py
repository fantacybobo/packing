"""Configuration and data structures for concave 3D model packing."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Pose:
    """A single pose: dx, dy, dz, qw, qx, qy, qz."""
    dx: float
    dy: float
    dz: float
    qw: float
    qx: float
    qy: float
    qz: float

    def as_array(self) -> np.ndarray:
        return np.array([self.dx, self.dy, self.dz, self.qw, self.qx, self.qy, self.qz])

    @staticmethod
    def from_array(arr) -> "Pose":
        return Pose(dx=arr[0], dy=arr[1], dz=arr[2],
                    qw=arr[3], qx=arr[4], qy=arr[5], qz=arr[6])


@dataclass
class ModelSpec:
    """Specification for a single STL model."""
    model_id: str
    stl_path: str
    poses: list  # List[Pose]


@dataclass
class PackingConfig:
    """Configuration for the concave packing operation.

    Attributes:
        area_x: (x_min, x_max) in world units (e.g., mm).
        area_y: (y_min, y_max) in world units (e.g., mm).
        margin: Gap between placed parts in world units.
        boundary_margin: Gap between parts and the area boundary in world units.
            Defaults to margin value if not set.
        bitmap_radix: Resolution of the occupancy bitmap (like Blender's bitmap_radix).
        max_placement_attempts: Max attempts per part before giving up.
        random_seed: Seed for reproducible random placement.
    """
    area_x: tuple = (-300.0, 300.0)
    area_y: tuple = (-200.0, 200.0)
    margin: float = 2.0
    boundary_margin: float = None
    bitmap_radix: int = 800
    max_placement_attempts: int = 2000
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.boundary_margin is None:
            self.boundary_margin = self.margin

    @property
    def area_width(self) -> float:
        return self.area_x[1] - self.area_x[0]

    @property
    def area_height(self) -> float:
        return self.area_y[1] - self.area_y[0]


@dataclass
class PlacementResult:
    """Result of placing a single part."""
    model_id: str
    pose_index: int
    rotation_quat: tuple      # (qw, qx, qy, qz) – the rotation applied
    translation: tuple         # (x, y, z) – final world position (dz from pose + packing position)
    placed: bool
    sample_x: float = 0.0     # packed x position in the area
    sample_y: float = 0.0     # packed y position in the area
