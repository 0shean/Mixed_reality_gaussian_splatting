#!/usr/bin/env python3
"""
Analyze coordinate bounds and distribution in PLY file
"""

import struct
import math

def analyze_ply_coords(ply_path):
    """Analyze the coordinate distribution in a PLY file."""

    print(f"Analyzing: {ply_path}\n")

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

        print(f"Vertices: {vertex_count:,}")
        print(f"Properties: {len(properties)}\n")

        # Read binary data
        floats_per_vertex = len(properties)
        bytes_per_vertex = floats_per_vertex * 4

        print("Reading position data...")
        positions = []

        for i in range(vertex_count):
            if i % 100000 == 0:
                print(f"  {i:,} / {vertex_count:,}", end='\r')

            vertex_data = f.read(bytes_per_vertex)
            floats = struct.unpack(f'{floats_per_vertex}f', vertex_data)
            positions.append(floats[0:3])  # x, y, z

        print(f"  {vertex_count:,} / {vertex_count:,} - Done!\n")

    # Calculate statistics
    min_pos = [min(p[i] for p in positions) for i in range(3)]
    max_pos = [max(p[i] for p in positions) for i in range(3)]
    mean_pos = [sum(p[i] for p in positions) / len(positions) for i in range(3)]
    center = [(min_pos[i] + max_pos[i]) / 2 for i in range(3)]
    extent = [max_pos[i] - min_pos[i] for i in range(3)]

    # Calculate std dev
    variance = [sum((p[i] - mean_pos[i])**2 for p in positions) / len(positions) for i in range(3)]
    std_pos = [math.sqrt(v) for v in variance]

    print("=" * 60)
    print("COORDINATE STATISTICS")
    print("=" * 60)
    print(f"           X          Y          Z")
    print(f"Min:   {min_pos[0]:9.4f}  {min_pos[1]:9.4f}  {min_pos[2]:9.4f}")
    print(f"Max:   {max_pos[0]:9.4f}  {max_pos[1]:9.4f}  {max_pos[2]:9.4f}")
    print(f"Mean:  {mean_pos[0]:9.4f}  {mean_pos[1]:9.4f}  {mean_pos[2]:9.4f}")
    print(f"Center:{center[0]:9.4f}  {center[1]:9.4f}  {center[2]:9.4f}")
    print(f"Extent:{extent[0]:9.4f}  {extent[1]:9.4f}  {extent[2]:9.4f}")
    print(f"Std:   {std_pos[0]:9.4f}  {std_pos[1]:9.4f}  {std_pos[2]:9.4f}")
    print()
    scene_size = math.sqrt(extent[0]**2 + extent[1]**2 + extent[2]**2)
    print(f"Scene size: {scene_size:.4f} units")

if __name__ == '__main__':
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else 'occam_bonsai_stripped.ply'
    analyze_ply_coords(filename)
