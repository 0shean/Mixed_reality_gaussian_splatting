#!/usr/bin/env python3
"""
Count how many Gaussians match a query above a similarity threshold.
"""

import struct
import sys

try:
    import torch
    import clip
    import numpy as np
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

def load_embeddings(embedding_file, num_gaussians):
    """Load OCCAM embeddings from binary file."""
    print(f"Loading embeddings from {embedding_file}...")

    with open(embedding_file, 'rb') as f:
        data = f.read()

    # Parse as 512D float32 vectors
    num_floats = len(data) // 4
    embeddings = struct.unpack(f'{num_floats}f', data)

    # Reshape into (num_gaussians, 512)
    embedding_dim = 512
    embeddings = [embeddings[i*embedding_dim:(i+1)*embedding_dim]
                  for i in range(num_gaussians)]

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✓ Loaded {len(embeddings)} embeddings of dimension {embedding_dim}")
    return embeddings

def count_matches(query, embeddings, threshold):
    """Count Gaussians matching query above threshold."""

    # Load CLIP model
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/16", device=device)
    print(f"✓ CLIP loaded on {device}\n")

    # Encode query
    print(f"Encoding query: '{query}'")
    query_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_embedding = model.encode_text(query_tokens)
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)[0]

    print(f"✓ Query encoded\n")

    # Normalize embeddings
    print("Computing similarities...")
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    gaussian_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute all similarities
    similarities = gaussian_norms @ query_norm

    # Count matches above threshold
    matches = similarities > threshold
    count = np.sum(matches)

    # Get statistics
    max_sim = np.max(similarities)
    min_sim = np.min(similarities)
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)

    print(f"✓ Analysis complete\n")
    print("=" * 60)
    print(f"RESULTS FOR QUERY: '{query}'")
    print("=" * 60)
    print(f"Threshold:           {threshold}")
    print(f"Matches:             {count:,} Gaussians ({count/len(embeddings)*100:.2f}%)")
    print(f"\nSimilarity Statistics:")
    print(f"  Max:               {max_sim:.6f}")
    print(f"  Mean:              {mean_sim:.6f}")
    print(f"  Median:            {median_sim:.6f}")
    print(f"  Min:               {min_sim:.6f}")

    # Show distribution
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    print(f"\nDistribution by threshold:")
    for t in thresholds:
        c = np.sum(similarities > t)
        print(f"  > {t:.2f}:  {c:>8,} Gaussians ({c/len(embeddings)*100:>5.2f}%)")

    # Top 10 matches
    print(f"\nTop 10 matching Gaussians:")
    top_indices = np.argsort(similarities)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. Gaussian #{idx:,} (similarity: {similarities[idx]:.6f})")

    return count, similarities

if __name__ == '__main__':
    if not HAS_DEPENDENCIES:
        print("ERROR: Missing dependencies!")
        print("\nRequired: torch, clip, numpy")
        sys.exit(1)

    # Configuration
    embedding_file = 'occam_bonsai_embeddings.bin'
    num_gaussians = 1332017

    query = "bonsai tree"
    threshold = 0.05

    # Load embeddings
    embeddings = load_embeddings(embedding_file, num_gaussians)

    # Count matches
    count, similarities = count_matches(query, embeddings, threshold)

    print("\n" + "=" * 60)
    print(f"✓ {count:,} Gaussians match '{query}' with similarity > {threshold}")
    print("=" * 60)
