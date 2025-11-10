#!/usr/bin/env python3
import sys
import numpy as np
from plyfile import PlyData, PlyElement
import struct


def split_ply_logits_to_bin(ply_path, out_ply_path, out_bin_path, logits_prefix="lang_feat_logit"):
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

    # Now I want to make sure all the logits have norm 1.
    # L2 normalization per row (each splat vector)
    # Avoids massive temporary allocations by computing in-place.
    print("Normalizing the splats")
    norms = np.linalg.norm(logits, axis=1, keepdims=True)
    # Replace zeros to avoid divide-by-zero
    np.maximum(norms, 1e-8, out=norms)
    logits /= norms

    # Write .bin file
    # Format:
    #   uint32 count
    #   float32[count * 512] logits (row-major)
    print(f"Writing binary logits file: {out_bin_path}")
    with open(out_bin_path, "wb") as f:
        # f.write(struct.pack("<I", num_vertices))  # little-endian uint32 count
        f.write(logits.tobytes(order="C"))  # raw float32 data

    # Remove logits from vertex attributes
    keep_fields = [f for f in all_props if f not in logits_fields]
    new_dtype = [(f, vertex_data.dtype[f]) for f in keep_fields]
    new_vertex_data = np.empty(num_vertices, dtype=new_dtype)
    for f in keep_fields:
        new_vertex_data[f] = vertex_data[f]

    # Write cleaned PLY
    new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([new_vertex_element], text=False).write(out_ply_path)
    print(f"  Wrote cleaned PLY to {out_ply_path}")

    print("✅ Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_ply_logits_bin.py input.ply [output_prefix]")
        sys.exit(1)

    input_ply = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else input_ply.rsplit('.', 1)[0]

    split_ply_logits_to_bin(
        ply_path=input_ply,
        out_ply_path=f"{prefix}_clean.ply",
        out_bin_path=f"{prefix}_logits.bin"
    )
