from fastapi import FastAPI, Body
from pydantic import BaseModel
import torch
import clip
import open_clip

# Load CLIP model
print('load clip model')
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use the same model that they use in the LangSplat paper.
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-16',
    pretrained='laion2b_s34b_b88k'
)
tokenizer = open_clip.get_tokenizer('ViT-B-16')

# Run the app with the command `uvicorn clip_server:app --host 0.0.0.0 --port 8000 --reload`
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/embed_text")
def embed_text(input: TextInput):
    print('embedding text:', input.text)
    with torch.no_grad():
        tokens = tokenizer([input.text]).to(device)
        embedding = model.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize
        return {"embedding": embedding[0].cpu().tolist()}
