"""
Embedding service with two backends:

  1. Gemini  (models/gemini-embedding-001, 3072 dims) — used when GEMINI_API_KEY is set
  2. Local   (all-MiniLM-L6-v2, 384 dims)             — used when no API key is available

Backend is selected once at startup based on settings.
Qdrant collection MUST be created with the matching EMBEDDING_DIM.
"""

import asyncio
from typing import List
from google import genai
from app.config.settings import get_settings
from app.utils.logger import logger

settings = get_settings()

# Dimensions produced by each backend
GEMINI_EMBEDDING_DIM = 3072
LOCAL_EMBEDDING_DIM  = 384

# Gemini supports up to 100 texts per embed_content call
_GEMINI_BATCH_SIZE = 100


def _gemini_available() -> bool:
    return bool(settings.GEMINI_API_KEY) and not settings.USE_VERTEX_AI


def _build_gemini_client() -> genai.Client:
    if settings.USE_VERTEX_AI:
        return genai.Client(
            vertexai=True,
            project=settings.VERTEX_PROJECT,
            location=settings.VERTEX_LOCATION,
        )
    return genai.Client(api_key=settings.GEMINI_API_KEY)


class EmbeddingService:
    def __init__(self):
        self._gemini_client = None
        self._local_model   = None
        self._use_gemini    = _gemini_available()

        backend = f"Gemini ({settings.EMBEDDING_MODEL}, dim={GEMINI_EMBEDDING_DIM})" \
                  if self._use_gemini else \
                  f"local all-MiniLM-L6-v2 (dim={LOCAL_EMBEDDING_DIM})"
        logger.info(f"Embedding backend: {backend}")

    # ── Client / model loaders ────────────────────────────────────────────────

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = _build_gemini_client()
        return self._gemini_client

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Local embedding model loaded: all-MiniLM-L6-v2")
        return self._local_model

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return GEMINI_EMBEDDING_DIM if self._use_gemini else LOCAL_EMBEDDING_DIM

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._use_gemini:
            try:
                return await self._embed_gemini(texts)
            except Exception as exc:
                logger.warning(f"Gemini embedding failed, falling back to local: {exc}")
                self._use_gemini = False
        return await self._embed_local(texts)

    async def embed_query(self, query: str) -> List[float]:
        results = await self.embed_texts([query])
        return results[0]

    # ── Backends ──────────────────────────────────────────────────────────────

    async def _embed_gemini(self, texts: List[str]) -> List[List[float]]:
        client      = self._get_gemini_client()
        all_vectors: List[List[float]] = []

        for i in range(0, len(texts), _GEMINI_BATCH_SIZE):
            batch    = texts[i:i + _GEMINI_BATCH_SIZE]
            response = await client.aio.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=batch,
            )
            all_vectors.extend([e.values for e in response.embeddings])

        return all_vectors

    async def _embed_local(self, texts: List[str]) -> List[List[float]]:
        loop  = asyncio.get_event_loop()
        model = self._get_local_model()
        return await loop.run_in_executor(
            None, lambda: model.encode(texts, normalize_embeddings=True).tolist()
        )


embedding_service = EmbeddingService()
