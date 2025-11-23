#!/usr/bin/env python3
"""
Filter out gaussians that lie to the left of the scene center.
This will remove approximately half of the gaussians based on x-coordinate.
"""

import numpy as np
from plyfile import PlyData, PlyElement
import sys

def filter_left_half(input_path: str, output_path: str):
    print(f"Reading PLY file: {input_path}")
    plydata = PlyData.read(input_path)
    vertex_data = plydata['vertex'].data

    total_gaussians = len(vertex_data)
    print(f"Total gaussians: {total_gaussians}")

    # Get x coordinates
    x_coords = vertex_data['x']

    # Calculate geometric center and median
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    x_geometric_center = (x_min + x_max) / 2.0
    x_median = np.median(x_coords)

    print(f"X range: [{x_min:.3f}, {x_max:.3f}]")
    print(f"X geometric center: {x_geometric_center:.3f}")
    print(f"X median (split point): {x_median:.3f}")

    # Create mask for gaussians on the right side (x >= median)
    # Using median ensures ~50/50 split
    mask = x_coords >= x_median

    kept_gaussians = np.sum(mask)
    removed_gaussians = total_gaussians - kept_gaussians

    print(f"Keeping {kept_gaussians} gaussians (right side)")
    print(f"Removing {removed_gaussians} gaussians (left side)")

    # Filter vertex data
    filtered_vertex_data = vertex_data[mask]

    # Create new PlyElement
    new_vertex_el = PlyElement.describe(filtered_vertex_data, 'vertex')

    # Create new PlyData with filtered vertices and other elements
    new_ply = PlyData(
        [new_vertex_el] + [e for e in plydata.elements if e.name != 'vertex'],
        text=plydata.text
    )

    # Write output
    print(f"Writing filtered PLY to: {output_path}")
    new_ply.write(output_path)
    print("Done!")

    # Print some statistics
    print("\nStatistics:")
    print(f"  Original: {total_gaussians:,} gaussians")
    print(f"  Filtered: {kept_gaussians:,} gaussians ({kept_gaussians/total_gaussians*100:.1f}%)")
    print(f"  Removed:  {removed_gaussians:,} gaussians ({removed_gaussians/total_gaussians*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_left_half.py <input.ply> <output.ply>")
        print("Example: python filter_left_half.py occam_bonsai.ply occam_bonsai_right_half.ply")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    filter_left_half(input_file, output_file)
