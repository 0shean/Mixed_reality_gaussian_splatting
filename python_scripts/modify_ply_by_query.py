#!/usr/bin/env python3
"""
Find Gaussians matching a query and modify the PLY file to highlight or hide them.
"""

import struct
import sys
import numpy as np

try:
    import torch
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

def load_embeddings(embedding_file, num_gaussians):
    """Load OCCAM embeddings from binary file."""
    print(f"Loading embeddings from {embedding_file}...")

    with open(embedding_file, 'rb') as f:
        data = f.read()

    num_floats = len(data) // 4
    embeddings = struct.unpack(f'{num_floats}f', data)

    embedding_dim = 512
    embeddings = [embeddings[i*embedding_dim:(i+1)*embedding_dim]
                  for i in range(num_gaussians)]

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✓ Loaded {len(embeddings)} embeddings\n")
    return embeddings

def find_matching_gaussians(query, embeddings, threshold):
    """Find indices of Gaussians matching query above threshold."""

    if not HAS_CLIP:
        print("ERROR: CLIP not available")
        return None

    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    print(f"✓ CLIP loaded on {device}\n")

    print(f"Encoding query: '{query}'")
    query_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_embedding = model.encode_text(query_tokens)
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)[0]

    print("Computing similarities...")
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    gaussian_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = gaussian_norms @ query_norm

    matching_indices = np.where(similarities > threshold)[0]
    matching_scores = similarities[matching_indices]

    print(f"✓ Found {len(matching_indices):,} matching Gaussians\n")

    return matching_indices, matching_scores

def modify_ply(input_ply, output_ply, matching_indices, mode='highlight'):
    """
    Modify PLY file based on matching indices.

    mode='highlight': Change color of matching Gaussians to bright yellow
    mode='hide': Set opacity of matching Gaussians to 0
    mode='remove': Remove matching Gaussians entirely
    mode='isolate': Keep only matching Gaussians, remove others
    """

    print(f"Reading PLY: {input_ply}")

    with open(input_ply, 'rb') as f:
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

        # Find property indices
        try:
            dc0_idx = properties.index('f_dc_0')
            dc1_idx = properties.index('f_dc_1')
            dc2_idx = properties.index('f_dc_2')
            opacity_idx = properties.index('opacity')
        except ValueError as e:
            print(f"ERROR: Missing required property - {e}")
            return

        # Read all data
        floats_per_vertex = len(properties)
        bytes_per_vertex = floats_per_vertex * 4

        print("Reading vertex data...")
        all_data = []
        for i in range(vertex_count):
            if i % 100000 == 0:
                print(f"  {i:,} / {vertex_count:,}", end='\r')

            vertex_data = f.read(bytes_per_vertex)
            floats = list(struct.unpack(f'{floats_per_vertex}f', vertex_data))
            all_data.append(floats)

        print(f"  {vertex_count:,} / {vertex_count:,} - Done!\n")

    # Convert matching indices to set for fast lookup
    matching_set = set(matching_indices)

    # Apply modifications
    print(f"Applying modifications (mode: {mode})...")
    modified_count = 0

    if mode == 'remove' or mode == 'isolate':
        # Filter vertices
        if mode == 'remove':
            filtered_data = [all_data[i] for i in range(len(all_data)) if i not in matching_set]
            print(f"Removed {len(matching_set):,} Gaussians")
        else:  # isolate
            filtered_data = [all_data[i] for i in range(len(all_data)) if i in matching_set]
            print(f"Kept only {len(matching_set):,} Gaussians")

        all_data = filtered_data
        vertex_count = len(all_data)

    else:
        # Modify in place
        for i in range(len(all_data)):
            if i % 100000 == 0:
                print(f"  {i:,} / {len(all_data):,}", end='\r')

            if i in matching_set:
                if mode == 'highlight':
                    # Change color to bright yellow (high f_dc values)
                    all_data[i][dc0_idx] = 1.0   # R
                    all_data[i][dc1_idx] = 1.0   # G
                    all_data[i][dc2_idx] = 0.0   # B
                elif mode == 'hide':
                    # Set opacity to 0
                    all_data[i][opacity_idx] = 0.0

                modified_count += 1

        print(f"  {len(all_data):,} / {len(all_data):,} - Done!")
        print(f"Modified {modified_count:,} Gaussians\n")

    # Write output
    print(f"Writing: {output_ply}")

    with open(output_ply, 'wb') as f:
        # Write header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {vertex_count}\n'.encode('ascii'))

        for prop in properties:
            f.write(f'property float {prop}\n'.encode('ascii'))

        f.write(b'end_header\n')

        # Write data
        for i, data in enumerate(all_data):
            if i % 100000 == 0:
                print(f"  {i:,} / {vertex_count:,}", end='\r')
            f.write(struct.pack(f'{floats_per_vertex}f', *data))

        print(f"  {vertex_count:,} / {vertex_count:,} - Done!")

    print(f"\n✓ Created: {output_ply}")

if __name__ == '__main__':
    # Configuration
    embedding_file = 'occam_bonsai_embeddings.bin'
    input_ply = 'occam_bonsai_centered.ply'
    num_gaussians = 1332017

    query = "bonsai tree"
    threshold = 0.05

    # Mode: 'highlight', 'hide', 'remove', or 'isolate'
    mode = 'highlight'  # Change this to your preference

    output_ply = f'occam_bonsai_{mode}_{query.replace(" ", "_")}.ply'

    print("=" * 60)
    print("GAUSSIAN QUERY-BASED PLY MODIFIER")
    print("=" * 60)
    print(f"Query:     '{query}'")
    print(f"Threshold: {threshold}")
    print(f"Mode:      {mode}")
    print(f"Input:     {input_ply}")
    print(f"Output:    {output_ply}")
    print("=" * 60)
    print()

    # Load embeddings
    embeddings = load_embeddings(embedding_file, num_gaussians)

    # Find matching Gaussians
    matching_indices, scores = find_matching_gaussians(query, embeddings, threshold)

    # Save indices to file
    indices_file = f'matching_indices_{query.replace(" ", "_")}.txt'
    print(f"Saving indices to: {indices_file}")
    with open(indices_file, 'w') as f:
        f.write(f"# Gaussians matching query: '{query}' (threshold > {threshold})\n")
        f.write(f"# Total matches: {len(matching_indices)}\n")
        f.write("#\n")
        f.write("# Index\tSimilarity\n")
        for idx, score in sorted(zip(matching_indices, scores), key=lambda x: x[1], reverse=True):
            f.write(f"{idx}\t{score:.6f}\n")
    print(f"✓ Saved {len(matching_indices):,} indices\n")

    # Modify PLY
    modify_ply(input_ply, output_ply, matching_indices, mode=mode)

    print("\n" + "=" * 60)
    print("✓ COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  1. {output_ply} - Modified PLY file")
    print(f"  2. {indices_file} - List of matching Gaussian indices")
    print()
    print("You can now import the modified PLY into Unity!")
