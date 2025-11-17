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

    num_vertices = len(vertex_data)
    feature_dim = 512  # Assuming 512-dim features

    features = np.zeros((num_vertices, feature_dim), dtype='float32')

    for i in range(num_vertices):
        for j in range(feature_dim):
            features[i, j] = vertex_data[i][f'lang_feat_logit_{j}']

    return features

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python occam_compute_scores.py <text_prompt> <output_file>")
        sys.exit(1)

    text_prompt = sys.argv[1]
    output_file = sys.argv[2]

    # Need to load occam features from a .bin file.

    computer = OccamScoreComputer()
    scores = computer.clip_embedding(text_prompt)

    known_prompts = ["bonsai forest", "a photo of a tree"]
    print("Known prompts: ", known_prompts)
    known_embeddings = [computer.clip_embedding(p) for p in known_prompts]

    known_array = np.vstack(known_embeddings)

    while True:
        raw_input = input("Enter prompt: ")
        text_embedding = computer.clip_embedding(raw_input)
        similarity = computer.compute_similarities(known_array, text_embedding)
        print(similarity)

    # Save scores to output file
    # scores.tofile(output_file)
    # print(f"Saved scores to {output_file}")
