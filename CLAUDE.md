# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install dependencies
uv run pytest tests/ -v          # Run all tests
uv run pytest tests/ -v -k TestConcavePacker  # Run a single test class
uv run python run_demo.py        # Run demo (generates packing visualizations)
uv run python tests/generate_test_data.py  # Regenerate STL test models
```

## Architecture

This is a **concave 3D model packing** system inspired by Blender's UV packing concave mode (`uv_pack.cc` / xatlas). Given STL models at specified poses (position + quaternion rotation), it packs them onto a 2D plane using bitmap occupancy for collision detection.

### Core Pipeline

1. **`stl_loader.py`** — Load STL via `trimesh`, apply quaternion rotation to mesh vertices
2. **`projection.py`** — Project 3D mesh → 2D by dropping Z, then union triangles via Shapely to get the exact **concave outline** (not convex hull). Ear-clipping triangulates the outline back into triangles for the packer.
3. **`pose_sampler.py`** — Randomly selects a pose (position + rotation) for each model from its `ModelSpec.poses` list
4. **`packer.py`** — The core: `OccupancyBitmap` rasterizes triangles into a boolean grid using edge-function scanline fill. `ConcavePacker.pack()` sorts shapes largest-first, finds non-overlapping positions via randomized grid search, and writes placed shapes into the bitmap. On placement failure, bitmap resolution doubles and placed shapes are re-traced for a retry.
5. **`visualize.py`** — Renders packing results as filled polygons with matplotlib, plus a concave-vs-convex comparison view

### Key Types (`config.py`)

- **`Pose`** — `(dx, dy, dz, qw, qx, qy, qz)` — position + quaternion rotation
- **`ModelSpec`** — model ID, STL path, list of valid poses
- **`PackingConfig`** — target area bounds, margin, bitmap resolution (`bitmap_radix`), placement attempts, random seed
- **`PlacementResult`** — final placement with translation, rotation, and placed/failed status

### Test Models

STL files live in `tests/test_data/`: L-shape, T-shape, rectangle, cylinder, small_square. The L-shape is the key model for testing concave packing — its notch creates space that convex-hull packing would waste.

### Important: CW Winding Convention

The `OccupancyBitmap` enforces **clockwise** winding for all triangles (matching Blender's convention). The `_ensure_cw_winding` static method is applied to all shapes before packing. Edge functions treat positive values as "inside" for CW triangles.
