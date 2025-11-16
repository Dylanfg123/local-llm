import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";

const app = express();
const VLLM_URL = process.env.VLLM_URL || "http://vllm:8000";
const RAG_URL  = process.env.RAG_URL  || "http://rag-api:7000";

// Debug: see EXACTLY what the browser asks for
app.use((req, _res, next) => { console.log(`[UI] ${req.method} ${req.url}`); next(); });

// Rewrite legacy paths BEFORE proxying
app.use((req, _res, next) => {
  if (req.url === "/models" || req.url.startsWith("/models/")) {
    req.url = req.url.replace(/^\/models/, "/v1/models");
  } else if (req.url === "/chat" || req.url.startsWith("/chat/")) {
    req.url = req.url.replace(/^\/chat/, "/v1/chat");
  }
  next();
});

// Serve ./public
app.use(express.static("./public"));

// Legacy aliases → redirect to v1 before proxying
app.all("/models", (req, res) => res.redirect(307, "/v1/models"));
app.all("/chat/completions", (req, res) => res.redirect(307, "/v1/chat/completions"));


// OpenAI-compatible API → vLLM
app.use("/v1", createProxyMiddleware({
  target: VLLM_URL,
  changeOrigin: true,
  ws: true,
  logLevel: "debug",
}));

// RAG passthrough
app.use("/rag", createProxyMiddleware({
  target: RAG_URL,
  changeOrigin: true,
  ws: true,
  logLevel: "debug",
}));

const PORT = process.env.PORT || 5173;
app.listen(PORT, () => console.log(`UI http://localhost:${PORT}`));
