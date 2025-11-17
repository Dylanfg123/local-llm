from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import os


MODEL_ID = os.getenv("MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI()
_model = SentenceTransformer(MODEL_ID)


class EmbedRequest(BaseModel):
    inputs: List[str]


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.inputs:
        return {"embeddings": []}

    vectors = _model.encode(req.inputs, batch_size=32, convert_to_numpy=False)
    return {"embeddings": [v.tolist() for v in vectors]}
