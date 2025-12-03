# Server
# 1. Gets a request with CLIP text
# 2. Computes the relevancy scores.

import argparse
import numpy as np
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
import io
from pydantic import BaseModel
import uvicorn

from occam_compute_scores import (
    OccamScoreComputer,
    load_occam_features_from_ply,
)

KNOWN_CANONICAL = "object"


class ServerState:
    """
    Holds all persistent server state:
    - CLIP computer
    - Canonical similarities
    - Language features from the PLY file
    """

    def __init__(self, ply_path: str):
        print("Initializing OccamScoreComputer...")
        self.computer = OccamScoreComputer()

        print(f"Loading PLY file: {ply_path}")
        self.occam_language_features = load_occam_features_from_ply(ply_path)

        print("Computing canonical similarity...")
        known_embeddings = self.computer.clip_embedding(KNOWN_CANONICAL)
        self.canonical_similarity = self.computer.compute_similarities(
            self.occam_language_features, known_embeddings
        )

        self.canonical_queries = [
            "object",
            "thing",
            "texture",
            "material"
        ]
        self.canonical_similarities = {}
        for query in self.canonical_queries:
            embedding = self.computer.clip_embedding(query)
            similarity = self.computer.compute_similarities(
                self.occam_language_features, embedding
            )
            self.canonical_similarities[query] = similarity

        print(
            "Canonical 'object' prompt similarities:",
            self.canonical_similarity.max(),
            self.canonical_similarity.min(),
            self.canonical_similarity.mean(),
        )


def get_state() -> ServerState:
    """
    We attach the state object to the FastAPI app during initialization.
    This dependency makes it accessible in each route.
    """
    return app.state.server_state


def compute_relevancy_scores(text: str, state: ServerState) -> np.ndarray:
    query_embedding = state.computer.clip_embedding(text)
    similarity = state.computer.compute_similarities(
        state.occam_language_features,
        query_embedding
    )

    # Compute softmax with each canonical similarity
    final_relevancy_score = np.ones(similarity.shape)  # to ensure shape consistency
    for query in state.canonical_similarities:
        canonical_similarity = state.canonical_similarities[query]
        relevancy_score = np.exp(similarity) / (np.exp(similarity) + np.exp(canonical_similarity))
        final_relevancy_score = np.minimum(final_relevancy_score, relevancy_score)

    return final_relevancy_score

# ---------------------
# FastAPI setup
# ---------------------

app = FastAPI()


class TextInput(BaseModel):
    text: str


@app.post("/relevancy_extrema")
def relevancy_extrema(
    input: TextInput,
    state: ServerState = Depends(get_state),
):
    print("embedding text:", input.text)

    # query_embedding = state.computer.clip_embedding(input.text)
    # similarity = state.computer.compute_similarities(
    #     state.occam_language_features, query_embedding
    # )

    # print(
    #     "Query prompt similarities:",
    #     similarity.max(),
    #     similarity.min(),
    #     similarity.mean(),
    # )

    relevancy_score = compute_relevancy_scores(input.text, state)

    min_relevancy = float(np.min(relevancy_score))
    max_relevancy = float(np.max(relevancy_score))

    return {
        "min_relevancy": min_relevancy,
        "max_relevancy": max_relevancy,
    }

@app.post("/similarity_binary")
def similarity_binary(
    input: TextInput,
    state: ServerState = Depends(get_state)
):
    print("embedding text:", input.text)

    query_embedding = state.computer.clip_embedding(input.text)
    similarity = state.computer.compute_similarities(
        state.occam_language_features,
        query_embedding
    )

    # Ensure float32 to minimize size
    buf = similarity.astype(np.float32).tobytes()

    # Wrap in a stream
    stream = io.BytesIO(buf)

    return StreamingResponse(
        stream,
        media_type="application/octet-stream"
    )

@app.post("/relevancy_score_binary")
def relevancy_score_binary(
    input: TextInput,
    state: ServerState = Depends(get_state)
):
    print("embedding text:", input.text)

    relevancy_score = compute_relevancy_scores(input.text, state)

    min_relevancy = float(np.min(relevancy_score))
    max_relevancy = float(np.max(relevancy_score))

    # normalized_relevancy = (relevancy_score - min_relevancy) / (max_relevancy - min_relevancy + 1e-8)

    # Ensure float32 to minimize size
    buf = relevancy_score.astype(np.float32).tobytes()

    # Wrap in a stream
    stream = io.BytesIO(buf)

    return StreamingResponse(
        stream,
        media_type="application/octet-stream"
    )



# ---------------------
# Main entry point
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIP server.")
    parser.add_argument("ply", type=str, help="Path to a .ply file to load")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Create and attach the server state
    app.state.server_state = ServerState(args.ply)

    # Start the server
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
