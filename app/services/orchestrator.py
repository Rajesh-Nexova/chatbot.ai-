import re
import time
from typing import List
from app.config.settings import get_settings
from app.models.schemas import (
    Intent, IntentResult, DocumentChunk, ChatRequest, ChatResponse,
    Citation, TokenUsage,
)
from app.services.intent_classifier import intent_classifier
from app.services.cache import cache_service
from app.services.llm_generator import llm_generator
from app.retrieval.vector_store import vector_store
from app.retrieval.embeddings import embedding_service
from app.retrieval.reranker import reranker
from app.utils.logger import logger
from app.utils.metrics import track_latency

settings = get_settings()


class RetrievalOrchestrator:

    async def handle(self, request: ChatRequest) -> ChatResponse:
        t_start = time.perf_counter()

        # ✅ SAFE CACHE READ
        try:
            cached = await cache_service.get("response", request.query)
        except Exception as e:
            logger.warning(f"Cache GET failed: {e}")
            cached = None

        if cached:
            logger.info("Cache HIT — returning cached response")
            cached["cached"] = True
            return ChatResponse(**cached)

        # ✅ SAFE Mongo history fetch
        conversation_history: List[dict] = []
        if request.session_id:
            try:
                from app.services.mongodb import mongo_service
                conversation_history = await mongo_service.get_conversation_history(
                    request.session_id
                )
            except Exception as e:
                logger.warning(f"Mongo fetch failed (continuing): {e}")

        # ✅ Intent classification
        async with track_latency("intent_classification"):
            intent_result: IntentResult = await intent_classifier.classify(request.query)

        logger.info(
            "Intent classified",
            extra={"intent": intent_result.intent, "confidence": intent_result.confidence},
        )

        answer = ""
        citations: List[Citation] = []
        token_usage: TokenUsage = None

        try:
            # 🚀 ROUTING LOGIC

            if intent_result.intent == Intent.WEB:
                async with track_latency("gemini_web_search"):
                    answer, citations, token_usage = await llm_generator.generate_with_search(
                        query=request.query,
                        conversation_history=conversation_history,
                    )

            elif intent_result.intent == Intent.DOMAIN:
                domain_chunks = await self._retrieve_domain(intent_result.rewritten_query)

                top_score = domain_chunks[0].score if domain_chunks else 0.0
                is_confident = (
                    bool(domain_chunks)
                    and top_score >= settings.SIMILARITY_THRESHOLD
                    and intent_result.confidence >= 0.60
                )

                if is_confident:
                    async with track_latency("llm_generation"):
                        answer, citations, token_usage = await llm_generator.generate(
                            query=request.query,
                            domain_chunks=domain_chunks,
                            conversation_history=conversation_history,
                        )
                else:
                    logger.info(
                        "Fallback to Gemini Search",
                        extra={"top_score": top_score, "chunks": len(domain_chunks)},
                    )
                    async with track_latency("gemini_search_fallback"):
                        answer, citations, token_usage = await llm_generator.generate_with_search(
                            query=request.query,
                            domain_chunks=domain_chunks,
                            conversation_history=conversation_history,
                        )

            else:  # GENERAL
                async with track_latency("llm_generation"):
                    answer, citations, token_usage = await llm_generator.generate(
                        query=request.query,
                        conversation_history=conversation_history,
                    )

        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"LLM pipeline failed: {e}")

        latency_ms = (time.perf_counter() - t_start) * 1000

        response = ChatResponse(
            answer=answer,
            intent=intent_result.intent,
            sources=citations,
            confidence=intent_result.confidence,
            latency_ms=round(latency_ms, 2),
            cached=False,
            token_usage=token_usage,
        )

        # ✅ SAFE CACHE WRITE
        try:
            await cache_service.set("response", request.query, response.model_dump())
        except Exception as e:
            logger.warning(f"Cache SET failed: {e}")

        # ✅ SAFE Mongo save
        if request.session_id:
            try:
                from app.services.mongodb import mongo_service
                from app.utils.encryption import encrypt

                await mongo_service.save_chat(
                    session_id=request.session_id,
                    encrypted_query=encrypt(request.query, settings.ENCRYPTION_KEY),
                    encrypted_response=encrypt(answer, settings.ENCRYPTION_KEY),
                    token_usage=token_usage.model_dump() if token_usage else None,
                )
            except Exception as e:
                logger.warning(f"Mongo save failed: {e}")

        return response

    # ───────────────────────────────────────────────
    # DOMAIN RETRIEVAL (SAFE VERSION)
    # ───────────────────────────────────────────────

    async def _retrieve_domain(self, query: str) -> List[DocumentChunk]:

        logger.info(f"Domain retrieval for query: '{query}'")
        original_query = query.strip()
        expanded_query = original_query
        if len(original_query.split()) <= 2:
            expanded_query = (
                f"{original_query} concepts explanation definition examples information"
            )
            logger.info(f"Expanded query: '{expanded_query}'")

        # ✅ SAFE CACHE READ
        try:
            cached_docs = await cache_service.get("docs", original_query)
            if cached_docs:
                logger.info(f"Cache hit: {len(cached_docs)} cached chunks")
                return [DocumentChunk(**d) for d in cached_docs]
        except Exception as e:
            logger.warning(f"Cache GET failed (docs): {e}")

        # ✅ EMBEDDING
        try:
            async with track_latency("embedding"):
                query_vector = await embedding_service.embed_query(original_query)
            logger.info(f"Query embedded successfully, vector length: {len(query_vector)}")
        except Exception as e:
            logger.error(f"Embedding failed: {e}", exc_info=True)
            return []

        # ✅ VECTOR SEARCH
        if not await vector_store.is_connected():
            logger.warning("Vector store not connected, skipping search")
            try:
                await vector_store.connect()
            except Exception as e:
                logger.error(f"❌ Failed to connect to vector store: {e}", exc_info=True)
                return []
        logger.info("✅ Successfully connected to vector store")
       
        
        try:
            
            async with track_latency("vector_search"):
                chunks = await vector_store.search_with_text(
                    query=original_query,
                    query_vector=query_vector,
                    top_k=settings.TOP_K * 2,
                )
            logger.info(f"Vector search returned {len(chunks)} chunks")
            if chunks:
                logger.info(f"Chunk scores: {[f'{c.score:.3f}' for c in chunks[:3]]}")
        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []

        if not chunks and expanded_query != original_query:
            logger.info("No relevant chunks from original query, retrying with expanded query")
            try:
                async with track_latency("embedding"):
                    expanded_vector = await embedding_service.embed_query(expanded_query)
                async with track_latency("vector_search"):
                    chunks = await vector_store.search_with_text(
                        query=expanded_query,
                        query_vector=expanded_vector,
                        top_k=settings.TOP_K * 2,
                    )
                logger.info(f"Expanded vector search returned {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Expanded query search failed: {e}", exc_info=True)
                return []

        # ✅ RERANK
        if chunks:
            try:
                async with track_latency("reranking"):
                    chunks = await reranker.rerank(
                        query, chunks, top_n=settings.RERANK_TOP_N
                    )
            except Exception as e:
                logger.warning(f"Reranker failed (continuing): {e}")

            # Filter chunks for specific queries to improve accuracy
            chunks = self._filter_chunks_by_relevance(query, chunks)

            # Clean the content of retrieved chunks to remove unwanted text
            from app.ingestion.cleaner import clean_text
            for chunk in chunks:
                chunk.content = clean_text(chunk.content)

            # ✅ SAFE CACHE WRITE
            try:
                await cache_service.set(
                    "docs", original_query, [c.model_dump() for c in chunks], ttl=600
                )
            except Exception as e:
                logger.warning(f"Cache SET failed (docs): {e}")

        return chunks

    def _filter_chunks_by_relevance(self, query: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Filter chunks to prioritize the most relevant sections for specific queries."""
        if not chunks:
            return chunks

        query_lower = query.lower().strip()

        # Define section mappings for specific queries
        section_mappings = {
            'headquarters': ['headquarters'],
            'headquarter': ['headquarters'],
            'location': ['headquarters'],
            'address': ['headquarters'],
            'office': ['headquarters'],
            'offices': ['headquarters'],
            'employees': ['employees'],
            'staff': ['employees'],
            'team': ['employees'],
            'people': ['employees'],
            'website': ['contact information'],
            'contact': ['contact information'],
            'email': ['contact information'],
            'phone': ['contact information'],
            'mission': ['mission and vision'],
            'vision': ['mission and vision'],
            'leadership': ['leadership'],
            'ceo': ['leadership'],
            'executive': ['leadership'],
            'management': ['leadership'],
            'services': ['core services'],
            'achievements': ['achievements and recognition'],
            'awards': ['achievements and recognition'],
            'recognition': ['achievements and recognition'],
            'certifications': ['achievements and recognition'],
        }

        # Check if query matches a specific section
        target_sections = []
        for keyword, sections in section_mappings.items():
            if keyword in query_lower:
                target_sections.extend(sections)
                break

        if target_sections:
            # Filter to only include chunks from target sections
            filtered_chunks = []
            for chunk in chunks:
                if chunk.section and any(target.lower() in chunk.section.lower() for target in target_sections):
                    filtered_chunks.append(chunk)
                    break  # Only take the first matching chunk for precision

            # If we found a matching section chunk, return only that
            if filtered_chunks:
                logger.info(f"Filtered to {len(filtered_chunks)} chunk(s) from section(s): {target_sections}")
                return filtered_chunks

            # Fallback: if section not found, choose the best chunk by content overlap
            query_words = set(re.findall(r"\w+", query_lower))
            fallback = []
            for chunk in chunks:
                content_words = set(re.findall(r"\w+", chunk.content.lower()))
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                if overlap > 0:
                    fallback.append((overlap, chunk))
            if fallback:
                fallback.sort(key=lambda x: x[0], reverse=True)
                logger.info("Fallback to best-matching chunk by content for specific query")
                return [fallback[0][1]]

        # For general queries, return top chunks but limit to avoid information overload
        return chunks[:min(3, len(chunks))]


orchestrator = RetrievalOrchestrator()