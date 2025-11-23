#!/usr/bin/env python3
"""
Create an Occam server that has the raw Occam feature vectors for all Gaussians.
It accepts a request with a string, then outputs the similarities for all Gaussians.
It saves its output in a file with N floats.
"""

import numpy as np
import torch
import open_clip

class OccamScoreComputer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained='laion2b_s34b_b88k'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')

    def clip_embedding(self, text_prompt):
        with torch.no_grad():
            tokens = self.tokenizer([text_prompt]).to(self.device)
            embedding = self.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize

        return embedding.cpu().numpy().astype('float32')

    def compute_similarities(self, occam_features, text_embedding):
        # occam_features is expected to be a 2D numpy array of shape (num_gaussians, feature_dim)
        occam_tensor = torch.from_numpy(occam_features).to(self.device)

        # text embedding is expected to be a 2D numpy array of shape (feature_dim, 1)
        embedding_tensor = torch.from_numpy(text_embedding).to(occam_tensor.device)

        dot_products = torch.matmul(occam_tensor, embedding_tensor.T).squeeze()
        return dot_products.cpu().numpy().astype('float32')


def load_occam_features_from_ply(path: str):
    from plyfile import PlyData

    plydata = PlyData.read(path)
    vertex_data = plydata['vertex'].data

    feature_dim = 512
    feature_fields = [f'lang_feat_logit_{j}' for j in range(feature_dim)]

    features = np.column_stack([vertex_data[field] for field in feature_fields]).astype('float32')

    print("Successfully loaded OCCAM features from PLY file, shape is", features.shape)

    return features

def turbo_colormap(values):
    """
    values: array in [0,1], shape (N,)
    returns: Nx3 RGB colors in [0,1]
    """

    # Polynomial coefficients from Google's Turbo colormap
    c0 = np.array([0.13572138, 0.09140261, 0.10667330], dtype=np.float32)
    c1 = np.array([0.47220780, 0.03405346, 0.82740960], dtype=np.float32)
    c2 = np.array([0.15588787, 2.21721700, 1.28103300], dtype=np.float32)
    c3 = np.array([-0.30102570,  -4.154547,   -2.4247400], dtype=np.float32)
    c4 = np.array([-0.00410288,   2.398376,    1.5770270], dtype=np.float32)
    c5 = np.array([0.0,          -0.7338740,  -0.7037280], dtype=np.float32)

    v = values.reshape(-1, 1)  # (N,1)

    colors = (
        c0
        + c1 * v
        + c2 * v**2
        + c3 * v**3
        + c4 * v**4
        + c5 * v**5
    )

    # Clamp to [0,1]
    return np.clip(colors, 0.0, 1.0)


def write_relevancy_to_ply(input_ply, output_ply, normalized_relevancy):
    from plyfile import PlyData, PlyElement
    """
    Writes the relevancy_scores in [0,1] to f_dc_0, f_dc_1, f_dc_2
    as a grayscale heatmap.
    """
    plydata = PlyData.read(input_ply)

    old_vertex = plydata['vertex'].data
    old_dtype = old_vertex.dtype

    # fields to remove
    remove_fields = {f'lang_feat_logit_{j}' for j in range(512)}

    # fields to KEEP (all non-lang fields)
    keep_fields = [name for name in old_dtype.names if name not in remove_fields]

    # build new dtype for the stripped vertex element
    new_dtype = [(name, old_dtype[name]) for name in keep_fields]

    # allocate new array
    new_vertex_data = np.empty(len(old_vertex), dtype=new_dtype)

    # copy over all preserved fields
    for name in keep_fields:
        new_vertex_data[name] = old_vertex[name]

    # ---- write heatmap colors ----
    colors = turbo_colormap(normalized_relevancy)

    # colors: Nx3 array in [0,1]
    new_vertex_data["f_dc_0"] = colors[:, 0]
    new_vertex_data["f_dc_1"] = colors[:, 1]
    new_vertex_data["f_dc_2"] = colors[:, 2]

    # ---- rebuild PLY elements ----

    new_vertex_el = PlyElement.describe(new_vertex_data, "vertex")

    # keep all non-vertex elements untouched
    other_elements = [e for e in plydata.elements if e.name != "vertex"]

    # build final PLY
    new_ply = PlyData(
        [new_vertex_el] + other_elements,
        text=plydata.text  # preserve ASCII/binary mode
    )

    # ---- write file ----
    new_ply.write(output_ply)
    print("Wrote cleaned + recolored PLY to:", output_ply)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python occam_compute_scores.py <input_ply_file> <output_file>")
        sys.exit(1)

    input_ply_file = sys.argv[1]
    output_file = sys.argv[2]

    # Need to load occam features from a .bin file.
    from plyfile import PlyData

    plydata = PlyData.read(input_ply_file)
    vertex_data = plydata['vertex'].data

    feature_dim = 512
    feature_fields = [f'lang_feat_logit_{j}' for j in range(feature_dim)]

    gaussian_occam_features = np.column_stack([vertex_data[field] for field in feature_fields]).astype('float32')

    print("Successfully loaded OCCAM features from PLY file, shape is", gaussian_occam_features.shape)

    computer = OccamScoreComputer()

    known_prompts = ["object"]
    known_embeddings = [computer.clip_embedding(p) for p in known_prompts]
    canonical_similarity = computer.compute_similarities(gaussian_occam_features, known_embeddings[0])
    print("Canonical 'object' prompt similarities:", canonical_similarity.max(), canonical_similarity.min(), canonical_similarity.mean())

    raw_input = input("Enter CLIP prompt: ")
    text_embedding = computer.clip_embedding(raw_input)
    similarity = computer.compute_similarities(gaussian_occam_features, text_embedding)
    print("Query prompt similarities:", similarity.max(), similarity.min(), similarity.mean())

    relevancy_score = np.exp(similarity) / (np.exp(similarity) + np.exp(canonical_similarity))
    print("Query prompt relevancy score:", relevancy_score.max(), relevancy_score.min(), relevancy_score.mean())

    normalized_scores = (relevancy_score - relevancy_score.min()) / (relevancy_score.max() - relevancy_score.min())
    # Change the color of the PLY vertices based on the normalized scores.
    # Color properties: "f_dc_0", "f_dc_1", "f_dc_2"
    write_relevancy_to_ply(input_ply_file, output_file, normalized_scores)
