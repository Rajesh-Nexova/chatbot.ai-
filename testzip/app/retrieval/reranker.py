from typing import List
import asyncio
from app.models.schemas import DocumentChunk
from app.config.settings import get_settings
from app.utils.logger import logger

settings = get_settings()

class Reranker:
    """
    Cross-encoder reranker. Uses sentence-transformers cross-encoder if available,
    otherwise falls back to score-based ordering.
    """
    def __init__(self):
        self._model = None
        self._available = False

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._available = True
            logger.info("Cross-encoder reranker loaded")
        except ImportError:
            logger.warning("sentence-transformers not installed — reranking disabled")
            self._available = False

    async def rerank(self, query: str, chunks: List[DocumentChunk], top_n: int = None) -> List[DocumentChunk]:
        top_n = top_n or settings.RERANK_TOP_N
        if not chunks:
            return chunks
        if len(chunks) <= top_n:
            return chunks

        if not self._available:
            self._load_model()

        if not self._available:
            return chunks[:top_n]

        loop = asyncio.get_event_loop()
        pairs = [[query, chunk.content] for chunk in chunks]
        scores = await loop.run_in_executor(
            None, lambda: self._model.predict(pairs).tolist()
        )

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda x: x.score, reverse=True)
        return reranked[:top_n]

reranker = Reranker()
