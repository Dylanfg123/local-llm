from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
import httpx, os, uuid

VLLM   = os.getenv("VLLM_URL",  "http://localhost:8000")
TEI    = os.getenv("TEI_URL",   "http://localhost:8080")
QDRANT = os.getenv("QDRANT_URL","http://localhost:6333")
API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "devkey123"))  # prefer VLLM_API_KEY
MODEL_ID = os.getenv("MODEL_ID", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")   # <- match /v1/models
COLL = "docs"

app = FastAPI()

async def embed(texts):
  async with httpx.AsyncClient(timeout=60) as cx:
    r = await cx.post(f"{TEI}/embed", json={"inputs": texts})
    r.raise_for_status()
    return r.json()["embeddings"]

async def ensure_collection():
  async with httpx.AsyncClient(timeout=30) as cx:
    await cx.put(f"{QDRANT}/collections/{COLL}", json={"vectors": {"size": 384, "distance": "Cosine"}})

@app.on_event("startup")
async def _start():
  await ensure_collection()
  # Optional: verify the model exists on vLLM
  try:
    async with httpx.AsyncClient(timeout=15) as cx:
      r = await cx.get(f"{VLLM}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
      r.raise_for_status()
      ids = {m["id"] for m in r.json().get("data", [])}
      if MODEL_ID not in ids:
        print(f"[warn] MODEL_ID '{MODEL_ID}' not served by vLLM; available: {ids}")
  except Exception as e:
    print(f"[warn] could not verify vLLM models: {e}")

@app.post("/ingest")
async def ingest(payload: dict = Body(...)):
  docs = payload["docs"]  # [{id?, text}]
  vecs = await embed([d["text"] for d in docs])
  points = [{"id": d.get("id", str(uuid.uuid4())), "vector": v, "payload": d} for d, v in zip(docs, vecs)]
  async with httpx.AsyncClient(timeout=60) as cx:
    r = await cx.put(f"{QDRANT}/collections/{COLL}/points?wait=true", json={"points": points})
    r.raise_for_status()
  return {"upserted": len(points)}

@app.post("/query")
async def query(payload: dict = Body(...)):
  messages = payload["messages"]
  user_msg = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
  k = payload.get("k", 5)
  vec = (await embed([user_msg]))[0]

  async with httpx.AsyncClient(timeout=60) as cx:
    sr = await cx.post(f"{QDRANT}/collections/{COLL}/points/search", json={"vector": vec, "limit": k})
    sr.raise_for_status()
    hits = sr.json()["result"]

  context = "\n\n".join([h["payload"]["text"] for h in hits])
  sys = "You are a helpful assistant. Use the CONTEXT when relevant. If unsure, say so."
  prompt_msgs = [
    {"role":"system","content":sys},
    {"role":"user","content": f"QUESTION:\n{user_msg}\n\nCONTEXT:\n{context}"}
  ]

  try:
    async with httpx.AsyncClient(timeout=None) as cx:
      r = await cx.post(
        f"{VLLM}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL_ID, "messages": prompt_msgs, "temperature": 0.2, "max_tokens": 512},
      )
      r.raise_for_status()
      return r.json()
  except httpx.HTTPStatusError as e:
    # Bubble up helpful info instead of a blank UI
    return JSONResponse(status_code=e.response.status_code, content={
      "error": "vllm_request_failed",
      "detail": e.response.text
    })
