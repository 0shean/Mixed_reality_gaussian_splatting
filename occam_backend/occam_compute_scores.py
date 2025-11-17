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

    def compute_similarity(self, text_prompt):
        with torch.no_grad():
            tokens = self.tokenizer([text_prompt]).to(self.device)
            embedding = self.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize

        return embedding.cpu().numpy().astype('float32')

    def compute_dot_products(self, text_embedding, occam_features):
        # occam_features is expected to be a 2D numpy array of shape (num_gaussians, feature_dim)
        occam_tensor = torch.from_numpy(occam_features).to(self.device)

        # text embedding is expected to be a 2D numpy array of shape (feature_dim, 1)
        embedding_tensor = torch.from_numpy(text_embedding).to(occam_tensor.device)

        dot_products = torch.matmul(occam_tensor, embedding_tensor).squeeze()
        return dot_products.cpu().numpy().astype('float32')

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python occam_compute_scores.py <text_prompt> <output_file>")
        sys.exit(1)

    text_prompt = sys.argv[1]
    output_file = sys.argv[2]

    # Need to load occam features from a .bin file.

    computer = OccamScoreComputer()
    scores = computer.compute_similarity(text_prompt)

    known = computer.compute_similarity("bonsai forest")
    known2 = computer.compute_similarity("a photo of a tree")

    known_stack = np.stack([known.squeeze(), known2.squeeze()], axis=1)

    similarity = computer.compute_dot_products(known_stack, scores)
    print(similarity)


    # Save scores to output file
    # scores.tofile(output_file)
    # print(f"Saved scores to {output_file}")
