#!/usr/bin/env python3
"""
Find the most similar words for each Gaussian's OCCAM embedding.
Uses CLIP text encoder to compare Gaussian embeddings against a vocabulary.
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

    print(f"✓ Loaded {len(embeddings)} embeddings of dimension {embedding_dim}")
    return np.array(embeddings, dtype=np.float32)

def find_top_words(gaussian_embedding, text_embeddings, vocabulary, top_k=10):
    """Find top-K words most similar to a Gaussian embedding."""

    # Normalize embeddings
    gaussian_norm = gaussian_embedding / np.linalg.norm(gaussian_embedding)
    text_norms = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # Cosine similarity
    similarities = text_norms @ gaussian_norm

    # Get top-K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_words = [(vocabulary[i], similarities[i]) for i in top_indices]

    return top_words

def main():
    if not HAS_DEPENDENCIES:
        print("ERROR: Missing dependencies!")
        print("\nPlease install required packages:")
        print("  pip install torch torchvision")
        print("  pip install git+https://github.com/openai/CLIP.git")
        print("  pip install numpy")
        sys.exit(1)

    # Configuration
    embedding_file = 'occam_bonsai_embeddings.bin'
    num_gaussians = 1332017

    # Sample vocabulary (you can expand this)
    vocabulary = [
        # Objects
        "table", "chair", "plant", "pot", "vase", "tree", "bonsai",
        "shelf", "wall", "floor", "ceiling", "window", "door",
        "book", "bottle", "cup", "bowl", "plate", "lamp", "light",
        "box", "container", "bag", "cloth", "fabric",

        # Materials
        "wood", "wooden", "metal", "plastic", "glass", "ceramic",
        "stone", "concrete", "paper", "cardboard",

        # Colors
        "red", "green", "blue", "yellow", "white", "black",
        "brown", "gray", "grey", "orange", "purple",

        # Descriptions
        "small", "large", "big", "tiny", "round", "square",
        "rectangular", "flat", "curved", "smooth", "rough",
        "shiny", "matte", "dark", "light", "bright", "dim",

        # Plant-specific (for bonsai scene)
        "leaf", "leaves", "branch", "branches", "trunk", "bark",
        "soil", "dirt", "roots", "foliage", "green plant",

        # Scene elements
        "background", "foreground", "shadow", "highlight",
        "surface", "edge", "corner", "center",

        # Combined descriptions
        "wooden table", "bonsai tree", "plant pot", "ceramic pot",
        "green leaves", "brown trunk", "small tree", "small plant",
        "potted plant", "indoor plant", "decorative plant",
    ]

    print("=" * 60)
    print("OCCAM Gaussian Embedding Decoder")
    print("=" * 60)
    print()

    # Load CLIP model
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✓ CLIP loaded on {device}")
    print()

    # Encode vocabulary
    print(f"Encoding {len(vocabulary)} words with CLIP...")
    text_tokens = clip.tokenize(vocabulary).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings = text_embeddings.cpu().numpy().astype(np.float32)
    print(f"✓ Text embeddings shape: {text_embeddings.shape}")
    print()

    # Load Gaussian embeddings
    gaussian_embeddings = load_embeddings(embedding_file, num_gaussians)
    print()

    # Analyze sample Gaussians
    print("=" * 60)
    print("TOP WORDS FOR SAMPLE GAUSSIANS")
    print("=" * 60)

    # Sample indices (evenly spaced)
    sample_indices = [
        0,                          # First Gaussian
        num_gaussians // 4,         # Quarter
        num_gaussians // 2,         # Middle
        3 * num_gaussians // 4,     # Three-quarters
        num_gaussians - 1,          # Last
    ]

    # You can also specify specific indices
    # sample_indices = [0, 100, 1000, 10000, 100000]

    for idx in sample_indices:
        print(f"\nGaussian #{idx:,}:")
        top_words = find_top_words(
            gaussian_embeddings[idx],
            text_embeddings,
            vocabulary,
            top_k=10
        )

        for rank, (word, score) in enumerate(top_words, 1):
            print(f"  {rank:2d}. {word:20s} (similarity: {score:.4f})")

    # Find Gaussians most similar to specific query words
    print("\n" + "=" * 60)
    print("GAUSSIANS MATCHING SPECIFIC QUERIES")
    print("=" * 60)

    queries = ["bonsai tree", "wooden table", "green leaves", "ceramic pot"]

    for query in queries:
        print(f"\nQuery: '{query}'")

        # Encode query
        query_tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            query_embedding = model.encode_text(query_tokens)
            query_embedding = query_embedding.cpu().numpy().astype(np.float32)[0]

        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        gaussian_norms = gaussian_embeddings / np.linalg.norm(gaussian_embeddings, axis=1, keepdims=True)

        # Compute similarities
        similarities = gaussian_norms @ query_norm

        # Top 5 Gaussians
        top_indices = np.argsort(similarities)[::-1][:5]

        print(f"  Top 5 matching Gaussians:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"    {rank}. Gaussian #{idx:,} (similarity: {similarities[idx]:.4f})")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nTo expand vocabulary, edit the 'vocabulary' list in this script.")
    print("You can add domain-specific words relevant to your scene.")

if __name__ == '__main__':
    main()
