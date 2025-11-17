#!/usr/bin/env python3
"""
Transform PLY coordinates - center and optionally rotate
"""

import struct
import math
import sys

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )

def axis_angle_to_quaternion(axis, angle_deg):
    """Convert axis-angle to quaternion."""
    angle_rad = math.radians(angle_deg)
    half_angle = angle_rad / 2
    s = math.sin(half_angle)
    c = math.cos(half_angle)

    # Normalize axis
    length = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    ax, ay, az = axis[0]/length, axis[1]/length, axis[2]/length

    return (c, ax*s, ay*s, az*s)

def rotate_vector(v, q):
    """Rotate vector v by quaternion q."""
    # Convert vector to quaternion (0, vx, vy, vz)
    v_quat = (0, v[0], v[1], v[2])

    # q_conjugate
    q_conj = (q[0], -q[1], -q[2], -q[3])

    # result = q * v * q_conjugate
    temp = quaternion_multiply(q, v_quat)
    result = quaternion_multiply(temp, q_conj)

    return (result[1], result[2], result[3])

def transform_ply(input_path, output_path, center=True, rotation=None):
    """
    Transform PLY coordinates.

    Args:
        input_path: Input PLY file
        output_path: Output PLY file
        center: If True, center coordinates at origin
        rotation: Dict with 'axis' (x,y,z) and 'angle' (degrees), or None
    """

    print(f"Reading: {input_path}")

    with open(input_path, 'rb') as f:
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
        print(f"Properties: {len(properties)}")

        # Find position and rotation indices
        try:
            pos_idx = [properties.index('x'), properties.index('y'), properties.index('z')]
            nor_idx = [properties.index('nx'), properties.index('ny'), properties.index('nz')]
            rot_idx = [properties.index('rot_0'), properties.index('rot_1'),
                      properties.index('rot_2'), properties.index('rot_3')]
        except ValueError as e:
            print(f"Error: Missing required property - {e}")
            return

        # Read all data
        floats_per_vertex = len(properties)
        bytes_per_vertex = floats_per_vertex * 4

        print("\nReading vertex data...")
        all_data = []
        for i in range(vertex_count):
            if i % 100000 == 0:
                print(f"  {i:,} / {vertex_count:,}", end='\r')

            vertex_data = f.read(bytes_per_vertex)
            floats = list(struct.unpack(f'{floats_per_vertex}f', vertex_data))
            all_data.append(floats)

        print(f"  {vertex_count:,} / {vertex_count:,} - Done!")

    # Calculate center offset
    offset = [0, 0, 0]
    if center:
        print("\nCalculating center...")
        positions = [[d[pos_idx[0]], d[pos_idx[1]], d[pos_idx[2]]] for d in all_data]
        min_pos = [min(p[i] for p in positions) for i in range(3)]
        max_pos = [max(p[i] for p in positions) for i in range(3)]
        offset = [(min_pos[i] + max_pos[i]) / 2 for i in range(3)]
        print(f"Center offset: ({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})")

    # Calculate rotation quaternion
    rot_quat = None
    if rotation:
        axis = rotation['axis']
        angle = rotation['angle']
        rot_quat = axis_angle_to_quaternion(axis, angle)
        print(f"\nRotation: {angle}° around axis ({axis[0]}, {axis[1]}, {axis[2]})")
        print(f"Quaternion: ({rot_quat[0]:.4f}, {rot_quat[1]:.4f}, {rot_quat[2]:.4f}, {rot_quat[3]:.4f})")

    # Apply transformations
    print("\nApplying transformations...")
    for i, data in enumerate(all_data):
        if i % 100000 == 0:
            print(f"  {i:,} / {vertex_count:,}", end='\r')

        # Get position
        pos = [data[pos_idx[0]], data[pos_idx[1]], data[pos_idx[2]]]

        # Center
        if center:
            pos = [pos[j] - offset[j] for j in range(3)]

        # Rotate position
        if rot_quat:
            pos = rotate_vector(pos, rot_quat)

        # Update position
        data[pos_idx[0]], data[pos_idx[1]], data[pos_idx[2]] = pos

        # Rotate normal
        if rot_quat:
            normal = [data[nor_idx[0]], data[nor_idx[1]], data[nor_idx[2]]]
            normal = rotate_vector(normal, rot_quat)
            data[nor_idx[0]], data[nor_idx[1]], data[nor_idx[2]] = normal

        # Rotate splat orientation quaternion
        if rot_quat:
            splat_rot = (data[rot_idx[0]], data[rot_idx[1]], data[rot_idx[2]], data[rot_idx[3]])
            # New rotation = rot_quat * splat_rot
            new_rot = quaternion_multiply(rot_quat, splat_rot)
            data[rot_idx[0]], data[rot_idx[1]], data[rot_idx[2]], data[rot_idx[3]] = new_rot

    print(f"  {vertex_count:,} / {vertex_count:,} - Done!")

    # Write output
    print(f"\nWriting: {output_path}")
    with open(output_path, 'wb') as f:
        # Write header
        for line in header_lines:
            f.write((line + '\n').encode('ascii'))

        # Write data
        for i, data in enumerate(all_data):
            if i % 100000 == 0:
                print(f"  {i:,} / {vertex_count:,}", end='\r')
            f.write(struct.pack(f'{floats_per_vertex}f', *data))

    print(f"  {vertex_count:,} / {vertex_count:,} - Done!")
    print(f"\n✓ Transformation complete!")

if __name__ == '__main__':
    # Configuration
    input_file = 'occam_bonsai_stripped.ply'
    output_file = 'occam_bonsai_centered.ply'

    # Center coordinates
    do_center = True

    # Optional rotation - uncomment and modify as needed
    # Examples:
    # rotation = {'axis': (1, 0, 0), 'angle': 90}   # 90° around X axis
    # rotation = {'axis': (0, 1, 0), 'angle': 180}  # 180° around Y axis
    # rotation = {'axis': (0, 0, 1), 'angle': -45}  # -45° around Z axis
    rotation = None  # No rotation by default

    if len(sys.argv) > 1:
        # Parse command line arguments if provided
        # Usage: python transform_ply.py input.ply output.ply [--no-center] [--rotate axis angle]
        print("Using command line arguments...")
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.ply', '_transformed.ply')

    transform_ply(input_file, output_file, center=do_center, rotation=rotation)

    print(f"\nTo apply rotation, edit the script and set:")
    print(f"  rotation = {{'axis': (x, y, z), 'angle': degrees}}")
