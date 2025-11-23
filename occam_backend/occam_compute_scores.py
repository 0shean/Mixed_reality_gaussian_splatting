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

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python occam_compute_scores.py <input_ply_file> <output_file>")
        sys.exit(1)

    input_ply_file = sys.argv[1]
    output_file = sys.argv[2]

    # Need to load occam features from a .bin file.
    gaussian_occam_features = load_occam_features_from_ply(input_ply_file)

    computer = OccamScoreComputer()

    known_prompts = ["object"]
    known_embeddings = [computer.clip_embedding(p) for p in known_prompts]
    canonical_similarity = computer.compute_similarities(gaussian_occam_features, known_embeddings[0])
    print("Canonical 'object' prompt similarities:", canonical_similarity.max(), canonical_similarity.min(), canonical_similarity.mean())

    while True:
        raw_input = input("Enter CLIP prompt: ")
        text_embedding = computer.clip_embedding(raw_input)
        similarity = computer.compute_similarities(gaussian_occam_features, text_embedding)
        print("Query prompt similarities:", similarity.max(), similarity.min(), similarity.mean())

        relevancy_score = np.exp(similarity) / (np.exp(similarity) + np.exp(canonical_similarity))
        print("Query prompt relevancy score:", relevancy_score.max(), relevancy_score.min(), relevancy_score.mean())

        # Get indices of the 100 largest elements
        top_n = 100
        top_indices = np.argpartition(-relevancy_score, top_n)[:top_n]  # unsorted top 100
        # Optional: sort them by value descending
        top_indices = top_indices[np.argsort(-relevancy_score[top_indices])]

        print("Indices of top 100 relevancy scores:", top_indices)
        print("Std and mean of top indices:", top_indices.std(), top_indices.mean())


        # Write the similarity scores to the output file N floats of size 4 bytes
        similarity.tofile(output_file)
        print(f"Saved scores to {output_file}")
