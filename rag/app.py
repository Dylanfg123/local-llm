from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import httpx, os, uuid, io, asyncio
from typing import List, Dict
from pypdf import PdfReader
from ddgs import DDGS

VLLM   = os.getenv("VLLM_URL",  "http://localhost:8000")
TEI    = os.getenv("TEI_URL",   "http://localhost:8080")
QDRANT = os.getenv("QDRANT_URL","http://localhost:6333")
API_KEY = os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "devkey123"))  # prefer VLLM_API_KEY
MODEL_ID = os.getenv("MODEL_ID", "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit")   # <- match /v1/models
COLL = "docs"

def _int_env(var: str, default: int) -> int:
  try:
    return int(os.getenv(var, default))
  except (TypeError, ValueError):
    return default

DEFAULT_WEB_RESULTS = max(1, min(8, _int_env("WEB_SEARCH_RESULTS", 3)))

app = FastAPI()


def read_text_from_upload(file: UploadFile, raw: bytes) -> str:
  name = (file.filename or "").lower()
  ctype = (file.content_type or "").lower()
  if name.endswith(".pdf") or "pdf" in ctype:
    try:
      reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
      raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")
    pages = []
    for page in reader.pages:
      try:
        txt = page.extract_text() or ""
      except Exception:
        txt = ""
      if txt.strip():
        pages.append(txt.strip())
    text = "\n\n".join(pages).strip()
    if not text:
      raise HTTPException(status_code=400, detail="PDF did not contain extractable text")
    return text

  try:
    return raw.decode("utf-8")
  except UnicodeDecodeError:
    return raw.decode("utf-8", errors="ignore")


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
  text = text.replace("\r\n", "\n").strip()
  if not text:
    return []
  chunks = []
  start = 0
  while start < len(text):
    end = min(len(text), start + max_chars)
    chunk = text[start:end].strip()
    if chunk:
      chunks.append(chunk)
    if end == len(text):
      break
    start = max(0, end - overlap)
  return chunks

async def web_search(query: str, max_results: int = DEFAULT_WEB_RESULTS) -> List[Dict[str, str]]:
  """Fetch web snippets via DuckDuckGo in a worker thread."""
  q = (query or "").strip()
  if not q:
    return []

  try:
    max_requested = int(max_results or DEFAULT_WEB_RESULTS)
  except (TypeError, ValueError):
    max_requested = DEFAULT_WEB_RESULTS
  max_results = max(1, min(8, max_requested))

  def _search():
    with DDGS(timeout=10) as ddgs:
      return list(ddgs.text(q, max_results=max_results))

  try:
    raw_results = await asyncio.to_thread(_search)
  except Exception as exc:
    print(f"[warn] web search failed: {exc}")
    return []

  hits = []
  for row in raw_results:
    hits.append({
      "title": (row.get("title") or "").strip(),
      "snippet": (row.get("body") or row.get("snippet") or "").strip(),
      "url": (row.get("href") or row.get("url") or "").strip(),
    })
  return hits

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
  use_rag = bool(payload.get("use_rag", True))
  use_web = bool(payload.get("use_web_search"))
  k = payload.get("k", 5)

  rag_context_chunks = []
  rag_hits = []
  rag_error = None
  if use_rag and user_msg:
    vec = (await embed([user_msg]))[0]

    try:
      async with httpx.AsyncClient(timeout=60) as cx:
        sr = await cx.post(
          f"{QDRANT}/collections/{COLL}/points/search",
          json={"vector": vec, "limit": k, "with_payload": True}
        )
        sr.raise_for_status()
        rag_hits = sr.json().get("result", [])
    except httpx.HTTPError as exc:
      rag_error = str(exc)
      print(f"[warn] qdrant search failed: {exc}")
      rag_hits = []

    for h in rag_hits:
      payload = h.get("payload") or {}
      text = payload.get("text")
      if text:
        rag_context_chunks.append(text)

  web_hits = []
  if use_web and user_msg:
    web_hits = await web_search(user_msg, max_results=payload.get("web_results", DEFAULT_WEB_RESULTS))

  context_sections = []
  if rag_context_chunks:
    context_sections.append("DOCUMENT CONTEXT:\n" + "\n\n".join(rag_context_chunks))
  if web_hits:
    formatted = []
    for idx, hit in enumerate(web_hits, start=1):
      snippet = hit["snippet"]
      url = hit["url"]
      title = hit["title"] or f"Result {idx}"
      formatted.append(f"[{idx}] {title}\n{snippet}\nURL: {url}".strip())
    context_sections.append("WEB SEARCH RESULTS:\n" + "\n\n".join(formatted))

  context = "\n\n-----\n\n".join(context_sections).strip()
  sys = "You are a helpful assistant. Use the CONTEXT and WEB SEARCH RESULTS sections when relevant. If unsure, say so. Cite URLs when your answer depends on them."
  prompt_msgs = [
    {"role":"system","content":sys},
    {"role":"user","content": f"QUESTION:\n{user_msg}\n\nCONTEXT:\n{context or '(no external context provided)'}"}
  ]

  try:
    async with httpx.AsyncClient(timeout=None) as cx:
      r = await cx.post(
        f"{VLLM}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL_ID, "messages": prompt_msgs, "temperature": 0.2, "max_tokens": 512},
      )
      r.raise_for_status()
      out = r.json()
      out["meta"] = {
        "rag_hits": len(rag_hits),
        "rag_error": rag_error,
        "web_search_used": bool(web_hits),
        "web_results": web_hits,
      }
      return out
  except httpx.HTTPStatusError as e:
    # Bubble up helpful info instead of a blank UI
    return JSONResponse(status_code=e.response.status_code, content={
      "error": "vllm_request_failed",
      "detail": e.response.text
    })


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
  raw = await file.read()
  if not raw:
    raise HTTPException(status_code=400, detail="Empty file")

  text = read_text_from_upload(file, raw)

  chunks = chunk_text(text)
  if not chunks:
    raise HTTPException(status_code=400, detail="File did not contain readable text")

  docs = [{"id": str(uuid.uuid4()), "text": chunk, "source": file.filename, "chunk": idx} for idx, chunk in enumerate(chunks)]
  vecs = await embed([d["text"] for d in docs])
  points = [{"id": d["id"], "vector": v, "payload": d} for d, v in zip(docs, vecs)]

  async with httpx.AsyncClient(timeout=60) as cx:
    r = await cx.put(f"{QDRANT}/collections/{COLL}/points?wait=true", json={"points": points})
    r.raise_for_status()

  return {"chunks": len(points)}
