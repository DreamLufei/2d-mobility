from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterable

from ..rag import VaspWikiRagService
from ..rag.wiki_sync import load_house_policy_documents
from ..runtime.database import is_postgres_uri
from .schemas import RetrievedEvidence

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_+\-.]+")


def _tokenize(text: str) -> list[str]:
    return [item.lower() for item in _TOKEN_PATTERN.findall(str(text or ""))]


class PolicyKnowledgeBase:
    def __init__(
        self,
        *,
        database_uri: str,
        embedding_model: str,
        embedding_base_url: str,
        embedding_api_key: str,
        qa_model: str,
        qa_base_url: str,
        qa_api_key: str,
        rag_top_k: int = 6,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        reindex_batch_size: int = 64,
        strict_rag: bool = True,
        fallback_documents: list[dict[str, object]] | None = None,
    ):
        self.strict_rag = bool(strict_rag)
        self.fallback_documents = list(fallback_documents or [])
        self.rag_service = (
            VaspWikiRagService(
                database_uri=database_uri,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
                qa_model=qa_model,
                qa_base_url=qa_base_url,
                qa_api_key=qa_api_key,
                rag_top_k=rag_top_k,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                reindex_batch_size=reindex_batch_size,
            )
            if is_postgres_uri(database_uri)
            else None
        )

    def retrieve(
        self,
        *,
        query: str,
        stage: str = "",
        top_k: int = 5,
        corpora: Iterable[str] | None = None,
    ) -> list[RetrievedEvidence]:
        if self.rag_service is None:
            if self.strict_rag:
                raise RuntimeError("policy_rag_unavailable_under_strict_mode")
            requested_corpora = {str(item) for item in list(corpora or []) if str(item)}
            query_tokens = set(_tokenize(query))
            stage_tokens = set(_tokenize(stage))
            ranked: list[tuple[float, dict[str, object]]] = []
            for document in self.fallback_documents:
                corpus = str(document.get("corpus") or "")
                if requested_corpora and corpus not in requested_corpora:
                    continue
                text = str(document.get("text") or "")
                title = str(document.get("title") or "")
                heading = str(document.get("heading") or "")
                tags = [str(item) for item in list(document.get("tags") or []) if str(item)]
                score = (
                    2.5 * len(query_tokens & set(_tokenize(title)))
                    + 1.8 * len(query_tokens & set(_tokenize(heading)))
                    + 1.1 * len(query_tokens & set(_tokenize(text)))
                    + 2.0 * len(stage_tokens & set(_tokenize(" ".join(tags))))
                )
                if score > 0:
                    ranked.append((score, document))
            ranked.sort(key=lambda item: item[0], reverse=True)
            return [
                RetrievedEvidence(
                    corpus=str(item.get("corpus") or "house_policy"),
                    source_id=str(item.get("source_id") or item.get("title") or "house-policy"),
                    title=str(item.get("title") or item.get("source_id") or "house-policy"),
                    url_or_path=str(item.get("url") or ""),
                    heading=str(item.get("heading") or ""),
                    stage=str(item.get("stage") or ""),
                    snippet=str(item.get("text") or "")[:420],
                    score=float(score),
                    tags=[str(tag) for tag in list(item.get("tags") or []) if str(tag)],
                )
                for score, item in ranked[: max(1, int(top_k or 5))]
            ]
        citations = self.rag_service.retrieve(query=query, stage=stage, top_k=top_k, corpora=corpora)
        return [
            RetrievedEvidence(
                corpus=item.corpus,
                source_id=item.source_id,
                title=item.title,
                chunk_id=item.chunk_id,
                revision_id=item.revision_id,
                url_or_path=item.url,
                heading=item.heading,
                stage=item.stage,
                snippet=item.snippet,
                score=item.score,
                tags=list(item.tags or []),
            )
            for item in citations
        ]


@lru_cache(maxsize=4)
def default_knowledge_base(*, house_corpus_path: str | None = None, wiki_corpus_path: str | None = None) -> PolicyKnowledgeBase:
    del wiki_corpus_path
    database_uri = os.environ.get("MOBILITY_DB_URI") or ""
    embedding_model = os.environ.get("EMBEDDING_MODEL") or ""
    strict_rag = str(os.environ.get("RAG_REQUIRED", "true") or "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    house_policy_path = house_corpus_path or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "corpus", "house_policy.json")
    )
    fallback_documents = [item.model_dump(mode="json") for item in load_house_policy_documents(house_policy_path)]
    if not database_uri or not embedding_model:
        if strict_rag:
            raise RuntimeError("RAG_REQUIRED=true but MOBILITY_DB_URI/EMBEDDING_MODEL is not fully configured")
        return PolicyKnowledgeBase(
            database_uri="memory://policy-kb",
            embedding_model=embedding_model or "test-embedding",
            embedding_base_url="",
            embedding_api_key="",
            qa_model="",
            qa_base_url="",
            qa_api_key="",
            strict_rag=False,
            fallback_documents=fallback_documents,
        )
    return PolicyKnowledgeBase(
        database_uri=database_uri,
        embedding_model=embedding_model,
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("LLM_BASE_URL") or "",
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("LLM_API_KEY") or "",
        qa_model=os.environ.get("WIKI_QA_MODEL") or os.environ.get("LLM_MODEL") or "",
        qa_base_url=os.environ.get("LLM_BASE_URL") or "",
        qa_api_key=os.environ.get("LLM_API_KEY") or "",
        rag_top_k=max(1, int(os.environ.get("RAG_TOP_K") or 6)),
        chunk_size=max(200, int(os.environ.get("RAG_CHUNK_SIZE") or 1200)),
        chunk_overlap=max(0, int(os.environ.get("RAG_CHUNK_OVERLAP") or 180)),
        reindex_batch_size=max(1, int(os.environ.get("RAG_REINDEX_BATCH_SIZE") or 64)),
        strict_rag=strict_rag,
        fallback_documents=fallback_documents,
    )
