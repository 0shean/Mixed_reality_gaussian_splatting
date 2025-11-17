#!/usr/bin/env python3
"""
Split OCCAM Gaussian Splat PLY file into two files:
1. embeddings.bin - Language embeddings only (512 floats per Gaussian)
2. stripped.ply - Standard Gaussian splat data without embeddings
"""

import struct
import sys
import os

def split_ply(input_path, output_embeddings, output_ply):
    """Split PLY file into embeddings and stripped Gaussian data."""

    print(f"Reading: {input_path}")

    with open(input_path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        header_size = f.tell()
        print(f"Header size: {header_size} bytes ({len(header_lines)} lines)")

        # Parse header to get vertex count and properties
        vertex_count = 0
        properties = []
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == 'element' and parts[1] == 'vertex':
                vertex_count = int(parts[2])
            elif len(parts) >= 3 and parts[0] == 'property' and parts[1] == 'float':
                properties.append(parts[2])

        print(f"Vertices: {vertex_count:,}")
        print(f"Properties per vertex: {len(properties)}")

        # Identify language embedding indices
        lang_indices = []
        other_properties = []
        for i, prop in enumerate(properties):
            if prop.startswith('lang_feat_logit_'):
                lang_indices.append(i)
            else:
                other_properties.append(prop)

        print(f"Language features: {len(lang_indices)} (indices {min(lang_indices)}-{max(lang_indices)})")
        print(f"Other properties: {len(other_properties)}")

        # Read binary data
        floats_per_vertex = len(properties)
        bytes_per_vertex = floats_per_vertex * 4
        total_data_bytes = vertex_count * bytes_per_vertex

        print(f"\nReading binary data...")
        print(f"  {floats_per_vertex} floats/vertex × {vertex_count:,} vertices = {total_data_bytes:,} bytes")

        data = f.read(total_data_bytes)
        if len(data) != total_data_bytes:
            raise ValueError(f"Expected {total_data_bytes} bytes, got {len(data)}")

        print(f"✓ Read {len(data):,} bytes")

    # Extract embeddings and other data
    print(f"\nExtracting data...")
    embeddings = []
    other_data = []

    for vertex_idx in range(vertex_count):
        if vertex_idx % 100000 == 0:
            print(f"  Processing vertex {vertex_idx:,} / {vertex_count:,}", end='\r')

        offset = vertex_idx * bytes_per_vertex
        vertex_floats = struct.unpack(f'{floats_per_vertex}f',
                                      data[offset:offset + bytes_per_vertex])

        # Extract language embeddings
        emb = [vertex_floats[i] for i in lang_indices]
        embeddings.extend(emb)

        # Extract other properties
        other = [vertex_floats[i] for i in range(floats_per_vertex) if i not in lang_indices]
        other_data.extend(other)

    print(f"  Processing vertex {vertex_count:,} / {vertex_count:,} - Done!")

    # Write embeddings binary file
    print(f"\nWriting embeddings: {output_embeddings}")
    with open(output_embeddings, 'wb') as f:
        f.write(struct.pack(f'{len(embeddings)}f', *embeddings))

    emb_size = os.path.getsize(output_embeddings)
    print(f"✓ Wrote {emb_size:,} bytes ({emb_size / 1024**2:.2f} MB)")

    # Write stripped PLY file
    print(f"\nWriting stripped PLY: {output_ply}")

    with open(output_ply, 'wb') as f:
        # Write header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {vertex_count}\n'.encode('ascii'))

        for prop in other_properties:
            f.write(f'property float {prop}\n'.encode('ascii'))

        f.write(b'end_header\n')

        # Write binary data
        f.write(struct.pack(f'{len(other_data)}f', *other_data))

    ply_size = os.path.getsize(output_ply)
    print(f"✓ Wrote {ply_size:,} bytes ({ply_size / 1024**2:.2f} MB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Original file:      {os.path.getsize(input_path):,} bytes ({os.path.getsize(input_path) / 1024**3:.2f} GB)")
    print(f"Embeddings file:    {emb_size:,} bytes ({emb_size / 1024**2:.2f} MB)")
    print(f"Stripped PLY:       {ply_size:,} bytes ({ply_size / 1024**2:.2f} MB)")
    print(f"Total after split:  {emb_size + ply_size:,} bytes ({(emb_size + ply_size) / 1024**3:.2f} GB)")
    print(f"\nEmbedding dimensions: {len(lang_indices)}")
    print(f"Gaussians: {vertex_count:,}")
    print(f"Properties in stripped PLY: {len(other_properties)}")
    print(f"\n✓ Stripped PLY is under 2GB limit: {ply_size < 2 * 1024**3}")

if __name__ == '__main__':
    print("Usage: python split_ply.py <input.ply>")
    print("  Creates: <input>_embeddings.bin and <input>_stripped.ply")


    input_file = 'occam_bonsai.ply'
    base_name = os.path.splitext(input_file)[0]
    embeddings_file = f"{base_name}_embeddings.bin"
    stripped_file = f"{base_name}_stripped.ply"

    split_ply(input_file, embeddings_file, stripped_file)
