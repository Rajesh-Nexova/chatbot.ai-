import json
import re
from typing import Optional
import google.generativeai as genai
from app.config.settings import get_settings
from app.models.schemas import Intent, IntentResult
from app.utils.logger import logger
from google import genai
from google.genai import types
settings = get_settings()

INTENT_SYSTEM_PROMPT = """You are a high-accuracy intent classification engine for a domain-specific chatbot.

Classify the user query into EXACTLY ONE of these three intents:

1. "domain" — The query is about domain-specific information found in internal documentation, product knowledge bases, or industry-specific content. Examples: company policies, service procedures, pricing tiers, product features.

2. "web" — The query requires current, real-time, or recent information from the internet. Examples: current news, stock prices, recent events, live data, today's date/time.

3. "general" — The query is about general knowledge, common facts, math, coding concepts, definitions, or historical information that can be answered from training data.

Return ONLY valid JSON in this exact format:
{
  "intent": "domain" | "web" | "general",
  "confidence": <float 0.0-1.0>,
  "rewritten_query": "<improved, retrieval-friendly query>"
}"""


def _build_client() -> genai.Client:
    if settings.USE_VERTEX_AI:
        return genai.Client(
            vertexai=True,
            project=settings.VERTEX_PROJECT,
            location=settings.VERTEX_LOCATION,
        )
    return genai.Client(api_key=settings.GEMINI_API_KEY)


class IntentClassifier:
    def __init__(self):
        self._client = _build_client()
        # Disable thinking (thinking_budget=0) so Gemini 2.5 Flash outputs
        # plain JSON without wrapping preamble text.
        _thinking = None
        try:
            _thinking = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass  # older SDK versions don't have ThinkingConfig

        cfg_kwargs: dict = dict(
            system_instruction=INTENT_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=512,
            response_mime_type="application/json",
        )
        if _thinking is not None:
            cfg_kwargs["thinking_config"] = _thinking
        self._config = types.GenerateContentConfig(**cfg_kwargs)

    async def classify(self, query: str, domain_keywords: Optional[list] = None) -> IntentResult:
        """Classify query intent using LLM. Falls back to heuristics on failure."""
        try:
            return await self._llm_classify(query, domain_keywords)
        except Exception as e:
            logger.warning(f"LLM classification failed, using heuristics: {e}")
            return self._heuristic_classify(query)

    async def _llm_classify(self, query: str, domain_keywords: Optional[list] = None) -> IntentResult:
        prompt = query
        if domain_keywords:
            prompt += f"\n\nDomain keywords that indicate 'domain' intent: {', '.join(domain_keywords)}"

        response = await self._client.aio.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=self._config,
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {raw}")

        data = json.loads(json_match.group())

        return IntentResult(
            intent=Intent(data["intent"]),
            confidence=float(data["confidence"]),
            rewritten_query=data["rewritten_query"],
        )

    def _heuristic_classify(self, query: str) -> IntentResult:
        """Lightweight heuristic fallback."""
        query_lower = query.lower()

        web_signals = ["latest", "today", "current", "now", "recent", "live", "breaking", "weather"]
        general_signals = ["what is", "who is", "define", "explain", "how does", "history of", "when was"]

        if any(s in query_lower for s in web_signals):
            return IntentResult(intent=Intent.WEB, confidence=0.75, rewritten_query=query)
        if any(s in query_lower for s in general_signals):
            return IntentResult(intent=Intent.GENERAL, confidence=0.70, rewritten_query=query)

        # Short conversational queries (< 8 words) are likely general/follow-up
        if len(query_lower.split()) < 8:
            return IntentResult(intent=Intent.GENERAL, confidence=0.60, rewritten_query=query)

        return IntentResult(intent=Intent.DOMAIN, confidence=0.60, rewritten_query=query)


intent_classifier = IntentClassifier()
