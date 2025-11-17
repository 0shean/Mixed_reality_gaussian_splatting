#!/usr/bin/env python3
"""
Check if the scene is planar or has unusual orientation
"""

import struct
import math

def check_orientation(ply_path):
    """Check scene orientation and planarity."""

    print(f"Analyzing orientation: {ply_path}\n")

    with open(ply_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        vertex_count = 0
        properties = []
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == 'element' and parts[1] == 'vertex':
                vertex_count = int(parts[2])
            elif len(parts) >= 3 and parts[0] == 'property' and parts[1] == 'float':
                properties.append(parts[2])

        # Read positions and normals
        floats_per_vertex = len(properties)
        bytes_per_vertex = floats_per_vertex * 4

        print(f"Reading {vertex_count:,} vertices...")
        positions = []
        normals = []

        for i in range(vertex_count):
            if i % 100000 == 0:
                print(f"  {i:,} / {vertex_count:,}", end='\r')

            vertex_data = f.read(bytes_per_vertex)
            floats = struct.unpack(f'{floats_per_vertex}f', vertex_data)
            positions.append((floats[0], floats[1], floats[2]))  # x, y, z
            normals.append((floats[3], floats[4], floats[5]))    # nx, ny, nz

        print(f"  {vertex_count:,} / {vertex_count:,} - Done!\n")

    # Analyze normal distribution
    print("=" * 60)
    print("NORMAL VECTOR ANALYSIS")
    print("=" * 60)

    # Calculate average normal direction
    avg_normal = [
        sum(n[0] for n in normals) / len(normals),
        sum(n[1] for n in normals) / len(normals),
        sum(n[2] for n in normals) / len(normals)
    ]

    # Normalize
    length = math.sqrt(sum(x**2 for x in avg_normal))
    if length > 0:
        avg_normal = [x / length for x in avg_normal]

    print(f"Average normal direction: ({avg_normal[0]:.4f}, {avg_normal[1]:.4f}, {avg_normal[2]:.4f})")

    # Find dominant axis
    abs_normal = [abs(x) for x in avg_normal]
    dominant_axis = abs_normal.index(max(abs_normal))
    axis_names = ['X', 'Y', 'Z']
    print(f"Dominant axis: {axis_names[dominant_axis]} ({abs_normal[dominant_axis]:.2%} alignment)")

    # Check how planar the scene is by looking at variance along each axis
    print("\n" + "=" * 60)
    print("COORDINATE DISTRIBUTION")
    print("=" * 60)

    for i, axis_name in enumerate(axis_names):
        coords = [p[i] for p in positions]
        min_c = min(coords)
        max_c = max(coords)
        mean_c = sum(coords) / len(coords)
        variance = sum((c - mean_c)**2 for c in coords) / len(coords)
        std_dev = math.sqrt(variance)

        print(f"{axis_name}-axis:")
        print(f"  Range: {min_c:.4f} to {max_c:.4f} (extent: {max_c - min_c:.4f})")
        print(f"  Mean: {mean_c:.4f}, Std Dev: {std_dev:.4f}")
        print(f"  Std/Extent ratio: {std_dev / (max_c - min_c) if (max_c - min_c) > 0 else 0:.4f}")

    # Check aspect ratio
    extents = [
        max(p[i] for p in positions) - min(p[i] for p in positions)
        for i in range(3)
    ]

    print(f"\nAspect ratios:")
    print(f"  X:Y:Z = {extents[0]:.2f}:{extents[1]:.2f}:{extents[2]:.2f}")
    print(f"  Normalized = {extents[0]/min(extents):.2f}:{extents[1]/min(extents):.2f}:{extents[2]/min(extents):.2f}")

    # Determine if scene is very flat
    min_extent = min(extents)
    max_extent = max(extents)
    if min_extent / max_extent < 0.3:
        thin_axis = extents.index(min_extent)
        print(f"\n⚠️  Scene is VERY FLAT along {axis_names[thin_axis]}-axis!")
        print(f"   This could make {axis_names[thin_axis]} rotations hard to see.")

if __name__ == '__main__':
    check_orientation('occam_bonsai_centered.ply')
