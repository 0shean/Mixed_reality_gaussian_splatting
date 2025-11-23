from plyfile import PlyData, PlyElement
import numpy as np

def remove_lang_feat_logits(input_path: str, output_path: str, feature_dim: int = 512):
    # Read original PLY
    plydata = PlyData.read(input_path)
    vertex_el = plydata['vertex']
    vertex_data = vertex_el.data

    # Fields to remove
    feature_fields = {f'lang_feat_logit_{j}' for j in range(feature_dim)}

    # New dtype excluding lang_feat fields
    new_dtype = [(name, vertex_data.dtype.fields[name][0])
                 for name in vertex_data.dtype.names
                 if name not in feature_fields]

    # Build cleaned structured array
    cleaned_vertex_data = np.empty(len(vertex_data), dtype=new_dtype)
    for name in cleaned_vertex_data.dtype.names:
        cleaned_vertex_data[name] = vertex_data[name]

    # Create new PlyElement and PlyData
    new_vertex_el = PlyElement.describe(cleaned_vertex_data, 'vertex')

    new_ply = PlyData(
        [new_vertex_el] +
        [e for e in plydata.elements if e.name != 'vertex'],  # keep other elements
        text=plydata.text
    )

    # Write output file
    new_ply.write(output_path)
    print(f"Successfully wrote cleaned PLY file to {output_path}. Removed {feature_dim} lang_feat_logit_* fields.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python split_ply_simple.py <input_ply_file> <output_ply_file>")
        sys.exit(1)

    input_ply_file = sys.argv[1]
    output_ply_file = sys.argv[2]

    remove_lang_feat_logits(input_ply_file, output_ply_file)
