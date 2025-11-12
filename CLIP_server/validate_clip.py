#!/usr/bin/env python3
import sys
import numpy as np
from plyfile import PlyData, PlyElement
import struct
import torch
import clip
import open_clip


def debug_clip_logits(ply_path, input_text, logits_prefix="lang_feat_logit"):
    # Load PLY
    print(f"Reading {ply_path} ...")
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex'].data
    num_vertices = len(vertex_data)
    print(f"  Vertices: {num_vertices}")

    # Find all langFeatLogits_* attributes
    all_props = vertex_data.dtype.names
    logits_fields = [f for f in all_props if f.startswith(logits_prefix)]
    logits_fields.sort(key=lambda x: int(x.split('_')[-1]))  # ensure correct order
    print(f"  Found {len(logits_fields)} logits attributes")

    if len(logits_fields) != 512:
        print(f"⚠️ Warning: expected 512 logits, found {len(logits_fields)}")

    first = logits_fields[0]
    last = logits_fields[-1]
    offset0 = vertex_data.dtype.fields[first][1]
    offset1 = vertex_data.dtype.fields[last][1] + vertex_data.dtype[last].itemsize
    stride = offset1 - offset0
    expected_size = len(logits_fields) * 4  # 4 bytes per float32
    actual_size = stride
    if actual_size != expected_size:
        print(f"⚠️ Logit fields not contiguous ({actual_size} vs expected {expected_size}). "
            "Falling back to slower copy method.")
        assert False
        # logits = np.empty((num_vertices, len(logits_fields)), dtype=np.float32)
        # for i, f in enumerate(logits_fields):
        #     logits[:, i] = vertex_data[f]

    raw = vertex_data.view(np.uint8).reshape(num_vertices, -1)
    logits = np.ndarray((num_vertices, len(logits_fields)), dtype=np.float32,
                        buffer=raw, offset=offset0, strides=(raw.strides[0], 4))
    logits_norm = logits / np.linalg.norm(logits, axis=1, keepdims=True)

    print("Shape of the lang features logits", logits.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # # Use the same model that they use in the LangSplat paper.
    # model, preprocess = clip.load("ViT-B/16", device=device)
    # with torch.no_grad():
    #     tokens = clip.tokenize([input_text]).to(device)
    #     embedding = model.encode_text(tokens)
    #     embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize
    # clip_embed = embedding[0].cpu().numpy()
    # print(f"CLIP embedding for \"{input_text}\" shape: {clip_embed.shape}")

    # open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")

    # similarity = np.dot(logits_norm, clip_embed)
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-16',
        pretrained='laion2b_s34b_b88k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-16')

    with torch.no_grad():
        tokens = tokenizer([input_text]).to(device)
        embedding = model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)

    clip_embed = embedding[0].cpu().numpy()
    print(f"CLIP embedding for \"{input_text}\" shape: {clip_embed.shape}")

    # print(f"Similarity shape: {similarity.shape}")
    # print(f"Max similarity: {similarity.max():.4f}, Min similarity: {similarity.min():.4f}")
    for word in ["bear", "object", "ground", "sky", "texture"]:
        with torch.no_grad():
            tokens = tokenizer([word]).to(device)
            emb = model.encode_text(tokens)
            emb /= emb.norm(dim=-1, keepdim=True)
            emb = emb[0].cpu().numpy()
            dots = np.dot(logits_norm, emb)
            print(f"{word:>6}: max similarity {dots.max():.4f}, min similarity {dots.min():.4f}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_ply_logits_bin.py input.ply [output_prefix]")
        sys.exit(1)

    input_ply = sys.argv[1]
    input_text = sys.argv[2] if len(sys.argv) > 2 else "bear"

    debug_clip_logits(
        ply_path=input_ply,
        input_text=input_text
    )
