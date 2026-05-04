import asyncio
from typing import List, Optional, AsyncGenerator
from google import genai
from google.genai import types
from app.config.settings import get_settings
from app.models.schemas import DocumentChunk, WebSearchResult, Citation, TokenUsage
from app.utils.logger import logger

settings = get_settings()

SYSTEM_PROMPT = """You are a helpful assistant. Answer user questions clearly and precisely.

RULES:
1. If domain context is provided below, answer from that context. Do not add information not present in it.
2. If the conversation history contains the answer (e.g. the user told you their name), use it.
3. If no context or history is relevant, use your general knowledge to answer helpfully.
4. If you are truly uncertain and neither context, history, nor general knowledge can answer, say so briefly.
5. Always be concise. Use bullet points or numbered lists where helpful.
6. Do NOT hallucinate facts, URLs, or details not present in the provided context."""


def _build_client() -> genai.Client:
    if settings.USE_VERTEX_AI:
        return genai.Client(
            vertexai=True,
            project=settings.VERTEX_PROJECT,
            location=settings.VERTEX_LOCATION,
        )
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _format_domain_context(chunks: List[DocumentChunk]) -> str:
    if not chunks:
        return ""
    parts = ["## Retrieved Domain Context\n"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk.source} — {chunk.url}]\n{chunk.content}\n")
    return "\n".join(parts)


def _build_rag_citations(chunks: List[DocumentChunk]) -> List[Citation]:
    citations = []
    seen: set = set()
    for chunk in chunks:
        if chunk.url not in seen:
            citations.append(Citation(
                title=chunk.source,
                url=chunk.url,
                snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            ))
            seen.add(chunk.url)
    return citations


def _extract_token_usage(response) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        thoughts_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
        total_tokens=getattr(meta, "total_token_count", 0) or 0,
    )


def _extract_grounding_citations(response) -> List[Citation]:
    """
    Parse Gemini grounding_metadata to build source citations.

    Gemini may populate sources in two places:
      1. grounding_chunks[].web  — title + URI  (most responses)
      2. search_entry_point.rendered_content — HTML with <a href="..."> chips
         (used when grounding_chunks is None but search still ran)
    """
    import re as _re
    citations: List[Citation] = []
    seen: set = set()
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return citations
        grounding = getattr(candidates[0], "grounding_metadata", None)
        if not grounding:
            return citations

        # Path 1: grounding_chunks (standard)
        for chunk in getattr(grounding, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web:
                uri   = getattr(web, "uri", None)
                title = getattr(web, "title", None) or uri
                if uri and uri not in seen:
                    citations.append(Citation(title=title, url=uri))
                    seen.add(uri)

        # Path 2: search_entry_point HTML chips (fallback)
        if not citations:
            sep = getattr(grounding, "search_entry_point", None)
            html = getattr(sep, "rendered_content", None) if sep else None
            if html:
                for href, label in _re.findall(
                    r'href="(https://vertexaisearch[^"]+)"[^>]*>([^<]+)', html
                ):
                    if href not in seen:
                        citations.append(Citation(title=label.strip(), url=href))
                        seen.add(href)

    except Exception as exc:
        logger.warning(f"Grounding citation extraction failed: {exc}")
    return citations


def _build_history(conversation_history: Optional[List[dict]]) -> List[types.Content]:
    history = []
    for msg in (conversation_history or [])[-6:]:
        role = "user" if msg["role"] == "user" else "model"
        history.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    return history


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "resource exhausted" in msg


class LLMGenerator:
    def __init__(self):
        self._client = _build_client()
        # Base config — no search tool; used for RAG + general responses
        self._base_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_TOKENS,
        )
        # Search config — attaches Google Search as a built-in tool
        self._search_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_TOKENS,
            tools=[self._build_search_tool()],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_search_tool(self) -> types.Tool:
        """
        Gemini API  → google_search tool (Dynamic Retrieval)
        Vertex AI   → google_search_retrieval tool
        """
        if settings.USE_VERTEX_AI:
            return types.Tool(
                google_search_retrieval=types.GoogleSearchRetrieval(
                    dynamic_retrieval_config=types.DynamicRetrievalConfig(
                        dynamic_threshold=0.6,
                    )
                )
            )
        return types.Tool(google_search=types.GoogleSearch())

    async def _generate_with_retry(self, contents: str, config=None):
        """generate_content with exponential back-off on 429 quota errors."""
        config = config or self._base_config
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(settings.LLM_MAX_RETRIES):
            try:
                return await self._client.aio.models.generate_content(
                    model=settings.LLM_MODEL,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                if _is_quota_error(exc) and attempt < settings.LLM_MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Gemini quota error (attempt {attempt + 1}), retrying in {wait}s: {exc}"
                    )
                    await asyncio.sleep(wait)
                    last_exc = exc
                else:
                    raise
        raise last_exc

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate(
        self,
        query: str,
        domain_chunks: Optional[List[DocumentChunk]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> tuple[str, List[Citation], TokenUsage]:
        """
        Pure RAG / general generation — no web search tool.
        Used when domain retrieval is confident or intent is 'general'.
        Returns (answer, citations, token_usage).
        """
        domain_chunks = domain_chunks or []
        context       = _format_domain_context(domain_chunks)
        user_message  = f"{context}\n\n## User Question\n{query}" if context else query

        history = _build_history(conversation_history)
        if history:
            chat = self._client.aio.chats.create(
                model=settings.LLM_MODEL,
                history=history,
                config=self._base_config,
            )
            response = await chat.send_message(user_message)
        else:
            response = await self._generate_with_retry(user_message)

        citations   = _build_rag_citations(domain_chunks)
        token_usage = _extract_token_usage(response)
        return response.text, citations, token_usage

    async def generate_with_search(
        self,
        query: str,
        domain_chunks: Optional[List[DocumentChunk]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> tuple[str, List[Citation], TokenUsage]:
        """
        Generation with Google Search grounding enabled.
        Used when intent is 'web' or domain retrieval is insufficient.
        Gemini calls Google Search internally — citations come from grounding_metadata.
        Returns (answer, citations, token_usage).
        """
        domain_chunks = domain_chunks or []
        context       = _format_domain_context(domain_chunks)
        user_message  = f"{context}\n\n## User Question\n{query}" if context else query

        history = _build_history(conversation_history)
        if history:
            chat = self._client.aio.chats.create(
                model=settings.LLM_MODEL,
                history=history,
                config=self._search_config,
            )
            response = await chat.send_message(user_message)
        else:
            response = await self._generate_with_retry(user_message, config=self._search_config)

        # For web intent, citations come from Gemini grounding metadata.
        # Also include any domain chunks that were passed as context.
        citations   = _extract_grounding_citations(response) or _build_rag_citations(domain_chunks)
        token_usage = _extract_token_usage(response)
        return response.text, citations, token_usage

    async def generate_raw(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        response_mime_type: Optional[str] = None,
    ) -> tuple[str, TokenUsage]:
        """Single-turn generation without search tool."""
        cfg_kwargs: dict = dict(temperature=temperature, max_output_tokens=max_output_tokens)
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if response_mime_type:
            cfg_kwargs["response_mime_type"] = response_mime_type

        config   = types.GenerateContentConfig(**cfg_kwargs)
        response = await self._generate_with_retry(prompt, config=config)
        return response.text, _extract_token_usage(response)

    async def generate_raw_with_search(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ) -> tuple[str, List[Citation], TokenUsage]:
        """Single-turn generation with Google Search grounding. Used for car weight queries."""
        cfg_kwargs: dict = dict(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tools=[self._build_search_tool()],
        )
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction

        config   = types.GenerateContentConfig(**cfg_kwargs)
        response = await self._generate_with_retry(prompt, config=config)
        citations   = _extract_grounding_citations(response)
        token_usage = _extract_token_usage(response)
        return response.text, citations, token_usage

    async def generate_stream(
        self,
        query: str,
        domain_chunks: Optional[List[DocumentChunk]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens — pure RAG / general, no search tool."""
        domain_chunks = domain_chunks or []
        context       = _format_domain_context(domain_chunks)
        user_message  = f"{context}\n\n## User Question\n{query}" if context else query

        history = _build_history(conversation_history)
        if history:
            chat   = self._client.aio.chats.create(
                model=settings.LLM_MODEL,
                history=history,
                config=self._base_config,
            )
            stream = await chat.send_message_stream(user_message)
        else:
            stream = await self._client.aio.models.generate_content_stream(
                model=settings.LLM_MODEL,
                contents=user_message,
                config=self._base_config,
            )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def generate_stream_with_search(
        self,
        query: str,
        domain_chunks: Optional[List[DocumentChunk]] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens with Google Search grounding enabled.
        Used when intent is 'web' or domain fallback is needed.
        """
        domain_chunks = domain_chunks or []
        context       = _format_domain_context(domain_chunks)
        user_message  = f"{context}\n\n## User Question\n{query}" if context else query

        history = _build_history(conversation_history)
        if history:
            chat   = self._client.aio.chats.create(
                model=settings.LLM_MODEL,
                history=history,
                config=self._search_config,
            )
            stream = await chat.send_message_stream(user_message)
        else:
            stream = await self._client.aio.models.generate_content_stream(
                model=settings.LLM_MODEL,
                contents=user_message,
                config=self._search_config,
            )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text


llm_generator = LLMGenerator()
