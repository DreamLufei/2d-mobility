from __future__ import annotations

import hashlib
import os
import requests
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..runtime.context import RuntimeContext
from ..runtime.database import is_postgres_uri, sqlalchemy_database_uri
from ..agents.llm_client import llm_request_guard
from .models import RagAnswer, RagCitation, RagQueryResponse, WikiChunk, WikiDocument
from .wiki_sync import DEFAULT_API_URL, DEFAULT_PAGE_TITLES, iter_all_page_titles, load_house_policy_documents, sync_wiki_documents


RAG_COLLECTION_NAME = "mobility_vasp_wiki_rag"

_DOCUMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_documents (
    corpus TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision_id TEXT NOT NULL DEFAULT '',
    content_sha TEXT NOT NULL,
    title TEXT NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    url TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (corpus, source_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_documents_corpus ON rag_documents(corpus);
CREATE INDEX IF NOT EXISTS idx_rag_documents_stage ON rag_documents(stage);
"""


class OpenRouterEmbeddings(Embeddings):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        dimensions: int | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "").rstrip("/")
        self.dimensions = dimensions
        self.timeout = max(1, int(timeout or 60))
        if not self.model:
            raise RuntimeError("OpenRouter embeddings require a model name.")
        if not self.api_key:
            raise RuntimeError("OpenRouter embeddings require an API key.")
        if not self.base_url:
            raise RuntimeError("OpenRouter embeddings require a base URL.")

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": inputs if len(inputs) != 1 else inputs[0],
        }
        if self.dimensions is not None:
            payload["dimensions"] = int(self.dimensions)
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        try:
            body = dict(response.json() or {})
        except Exception as exc:
            raise ValueError(
                f"OpenRouter embedding request returned non-JSON response for model={self.model!r} "
                f"(status={response.status_code})."
            ) from exc
        error = dict(body.get("error") or {})
        if error:
            raise ValueError(
                f"OpenRouter embedding request failed for model={self.model!r}: "
                f"{error.get('message') or body} (code={error.get('code')}, http_status={response.status_code})"
            )
        data = list(body.get("data") or [])
        embeddings: list[list[float]] = []
        for item in data:
            vector = list((item or {}).get("embedding") or [])
            if not vector:
                raise ValueError(f"OpenRouter embedding request returned empty embedding for model={self.model!r}.")
            embeddings.append([float(value) for value in vector])
        if not embeddings:
            raise ValueError(f"OpenRouter embedding request returned no embedding data for model={self.model!r}.")
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([str(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([str(text)])[0]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _snippet(text: str, *, limit: int = 420) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


class VaspWikiRagService:
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
        collection_name: str = RAG_COLLECTION_NAME,
    ) -> None:
        self.database_uri = str(database_uri or "").strip()
        self.embedding_model = str(embedding_model or "").strip()
        self.embedding_base_url = str(embedding_base_url or "").strip()
        self.embedding_api_key = str(embedding_api_key or "").strip()
        self.qa_model = str(qa_model or "").strip()
        self.qa_base_url = str(qa_base_url or "").strip()
        self.qa_api_key = str(qa_api_key or "").strip()
        self.rag_top_k = max(1, int(rag_top_k or 6))
        self.chunk_size = max(200, int(chunk_size or 1200))
        self.chunk_overlap = max(0, int(chunk_overlap or 180))
        self.reindex_batch_size = max(1, int(reindex_batch_size or 64))
        self.collection_name = str(collection_name or RAG_COLLECTION_NAME)
        if not is_postgres_uri(self.database_uri):
            raise RuntimeError("VASP Wiki RAG requires a Postgres MOBILITY_DB_URI.")

    @classmethod
    def from_runtime(cls, runtime: RuntimeContext) -> "VaspWikiRagService":
        return cls(
            database_uri=runtime.resolved_db_uri,
            embedding_model=runtime.embedding_model,
            embedding_base_url=runtime.embedding_base_url,
            embedding_api_key=runtime.embedding_api_key,
            qa_model=runtime.wiki_qa_model or runtime.agent_runtime.llm_model,
            qa_base_url=runtime.agent_runtime.llm_base_url or "",
            qa_api_key=runtime.agent_runtime.llm_api_key or "",
            rag_top_k=runtime.rag_top_k,
            chunk_size=runtime.rag_chunk_size,
            chunk_overlap=runtime.rag_chunk_overlap,
            reindex_batch_size=runtime.rag_reindex_batch_size,
        )

    def _connect(self):
        return connect(self.database_uri, row_factory=dict_row, autocommit=True)

    def _embeddings(self) -> Embeddings:
        if "openrouter.ai" in self.embedding_base_url.lower():
            return OpenRouterEmbeddings(
                model=self.embedding_model,
                api_key=self.embedding_api_key,
                base_url=self.embedding_base_url,
            )
        return OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url or None,
        )

    def _chat_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.qa_model,
            api_key=self.qa_api_key,
            base_url=self.qa_base_url or None,
            temperature=0.0,
            max_tokens=1800,
        )

    def _vector_store(self, *, pre_delete_collection: bool = False) -> PGVector:
        return PGVector(
            embeddings=self._embeddings(),
            connection=sqlalchemy_database_uri(self.database_uri),
            collection_name=self.collection_name,
            use_jsonb=True,
            create_extension=True,
            pre_delete_collection=pre_delete_collection,
        )

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_DOCUMENT_SCHEMA)

    def upsert_documents(self, documents: Iterable[WikiDocument]) -> dict[str, int]:
        self.ensure_schema()
        inserted = 0
        updated = 0
        skipped = 0
        with self._connect() as conn:
            for document in documents:
                existing = conn.execute(
                    "SELECT revision_id, content_sha FROM rag_documents WHERE corpus=%s AND source_id=%s",
                    [document.corpus, document.source_id],
                ).fetchone()
                if existing and str(existing.get("revision_id") or "") == document.revision_id and str(existing.get("content_sha") or "") == document.content_sha:
                    skipped += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO rag_documents (
                        corpus, source_id, revision_id, content_sha, title, heading, stage, tags, url, text, metadata, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (corpus, source_id) DO UPDATE SET
                        revision_id=EXCLUDED.revision_id,
                        content_sha=EXCLUDED.content_sha,
                        title=EXCLUDED.title,
                        heading=EXCLUDED.heading,
                        stage=EXCLUDED.stage,
                        tags=EXCLUDED.tags,
                        url=EXCLUDED.url,
                        text=EXCLUDED.text,
                        metadata=EXCLUDED.metadata,
                        updated_at=NOW()
                    """,
                    [
                        document.corpus,
                        document.source_id,
                        document.revision_id,
                        document.content_sha,
                        document.title,
                        document.heading,
                        document.stage,
                        Jsonb(list(document.tags or [])),
                        document.url,
                        document.text,
                        Jsonb(dict(document.metadata or {})),
                    ],
                )
                if existing:
                    updated += 1
                else:
                    inserted += 1
        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def load_documents(self, *, corpora: Iterable[str] | None = None) -> list[WikiDocument]:
        self.ensure_schema()
        corpora_values = [str(item) for item in list(corpora or []) if str(item)]
        query = "SELECT corpus, source_id, revision_id, content_sha, title, heading, stage, tags, url, text, metadata FROM rag_documents"
        params: list[Any] = []
        if corpora_values:
            query += " WHERE corpus = ANY(%s)"
            params.append(corpora_values)
        query += " ORDER BY corpus ASC, source_id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            WikiDocument(
                corpus=str(row.get("corpus") or ""),
                source_id=str(row.get("source_id") or ""),
                revision_id=str(row.get("revision_id") or ""),
                content_sha=str(row.get("content_sha") or ""),
                title=str(row.get("title") or ""),
                heading=str(row.get("heading") or ""),
                stage=str(row.get("stage") or ""),
                tags=[str(tag) for tag in list(row.get("tags") or []) if str(tag)],
                url=str(row.get("url") or ""),
                text=str(row.get("text") or ""),
                metadata=dict(row.get("metadata") or {}),
            )
            for row in rows
        ]

    def _chunk_documents(self, documents: list[WikiDocument]) -> list[WikiChunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: list[WikiChunk] = []
        for document in documents:
            parts = splitter.split_text(document.text)
            for index, part in enumerate(parts):
                chunk_id = hashlib.sha1(
                    f"{document.corpus}:{document.source_id}:{document.revision_id}:{index}:{part}".encode("utf-8")
                ).hexdigest()
                chunks.append(
                    WikiChunk(
                        chunk_id=chunk_id,
                        corpus=document.corpus,
                        source_id=document.source_id,
                        revision_id=document.revision_id,
                        content_sha=document.content_sha,
                        title=document.title,
                        heading=document.heading,
                        stage=document.stage,
                        tags=list(document.tags or []),
                        url=document.url,
                        text=part,
                        sequence=index,
                        metadata=dict(document.metadata or {}),
                    )
                )
        return chunks

    def rebuild_index(self, *, corpora: Iterable[str] | None = None) -> dict[str, Any]:
        documents = self.load_documents(corpora=corpora)
        chunks = self._chunk_documents(documents)
        vector_store = self._vector_store(pre_delete_collection=True)
        if chunks:
            for start in range(0, len(chunks), self.reindex_batch_size):
                batch = chunks[start : start + self.reindex_batch_size]
                vector_store.add_texts(
                    [item.text for item in batch],
                    metadatas=[
                        {
                            "corpus": item.corpus,
                            "source_id": item.source_id,
                            "chunk_id": item.chunk_id,
                            "revision_id": item.revision_id,
                            "content_sha": item.content_sha,
                            "title": item.title,
                            "heading": item.heading,
                            "stage": item.stage,
                            "tags": list(item.tags or []),
                            "url": item.url,
                            "sequence": item.sequence,
                            **dict(item.metadata or {}),
                        }
                        for item in batch
                    ],
                    ids=[item.chunk_id for item in batch],
                )
        return {
            "status": "completed",
            "collection_name": self.collection_name,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "updated_at": _utc_now_iso(),
        }

    def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        corpora: Iterable[str] | None = None,
        stage: str = "",
    ) -> list[RagCitation]:
        resolved_top_k = max(1, int(top_k or self.rag_top_k))
        vector_store = self._vector_store()
        results = vector_store.similarity_search_with_relevance_scores(query, k=max(resolved_top_k * 4, 20))
        requested_corpora = {str(item) for item in list(corpora or []) if str(item)}
        requested_stage = str(stage or "").strip()
        citations: list[RagCitation] = []
        for document, score in results:
            metadata = dict(document.metadata or {})
            corpus = str(metadata.get("corpus") or "")
            if requested_corpora and corpus not in requested_corpora:
                continue
            tags = [str(item) for item in list(metadata.get("tags") or []) if str(item)]
            doc_stage = str(metadata.get("stage") or "")
            if requested_stage and requested_stage not in {doc_stage, *tags}:
                continue
            citations.append(
                RagCitation(
                    corpus=corpus,
                    source_id=str(metadata.get("source_id") or ""),
                    chunk_id=str(metadata.get("chunk_id") or ""),
                    revision_id=str(metadata.get("revision_id") or ""),
                    title=str(metadata.get("title") or metadata.get("source_id") or ""),
                    heading=str(metadata.get("heading") or ""),
                    url=str(metadata.get("url") or ""),
                    snippet=_snippet(document.page_content),
                    score=float(score or 0.0),
                    stage=doc_stage,
                    tags=tags,
                )
            )
            if len(citations) >= resolved_top_k:
                break
        return citations

    def answer_query(
        self,
        *,
        query: str,
        top_k: int | None = None,
        corpora: Iterable[str] | None = None,
        stage: str = "",
    ) -> RagAnswer:
        citations = self.retrieve(query=query, top_k=top_k, corpora=corpora, stage=stage)
        if not citations:
            return RagAnswer(
                query=query,
                answer="No relevant VASP Wiki evidence was found for this query.",
                citations=[],
                retrieval_metadata={"collection_name": self.collection_name, "top_k": int(top_k or self.rag_top_k)},
            )
        context = "\n\n".join(
            [
                f"[{index}] {item.title} | heading={item.heading} | url={item.url}\n{item.snippet}"
                for index, item in enumerate(citations, start=1)
            ]
        )
        with llm_request_guard(self.runtime.agent_runtime, role="specialist"):
            response = self._chat_model().invoke(
                [
                    (
                        "system",
                        "You answer VASP questions using only the retrieved context. Be concise, technical, and cite sources by title.",
                    ),
                    (
                        "human",
                        f"Question:\n{query}\n\nRetrieved context:\n{context}\n\nAnswer with short inline citations by title.",
                    ),
                ]
            )
        return RagAnswer(
            query=query,
            answer=str(getattr(response, "content", "") or "").strip(),
            citations=citations,
            retrieval_metadata={"collection_name": self.collection_name, "top_k": int(top_k or self.rag_top_k)},
        )

    def query(self, *, query: str, top_k: int | None = None, corpora: Iterable[str] | None = None, stage: str = "") -> RagQueryResponse:
        answer = self.answer_query(query=query, top_k=top_k, corpora=corpora, stage=stage)
        return RagQueryResponse(
            query=answer.query,
            answer=answer.answer,
            citations=answer.citations,
            retrieval_metadata=answer.retrieval_metadata,
        )

    def health(self) -> dict[str, Any]:
        self.ensure_schema()
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM rag_documents").fetchone() or {}
        return {
            "status": "ok",
            "database_backend": "postgres",
            "collection_name": self.collection_name,
            "document_count": int(row.get("count") or 0),
        }

    def sync(
        self,
        *,
        mode: str = "incremental",
        api_url: str = DEFAULT_API_URL,
        include_all_pages: bool = False,
        max_pages: int | None = None,
        delay_seconds: float = 0.2,
        titles: list[str] | None = None,
        house_policy_path: str | None = None,
    ) -> dict[str, Any]:
        del mode
        target_titles = list(titles or [])
        if include_all_pages:
            target_titles.extend(iter_all_page_titles(api_url, max_pages=max_pages))
        if not target_titles:
            target_titles.extend(DEFAULT_PAGE_TITLES if max_pages is None else DEFAULT_PAGE_TITLES[:max_pages])
        house_path = house_policy_path or os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "policy", "corpus", "house_policy.json")
        )
        house_documents = load_house_policy_documents(house_path)
        wiki_documents = sync_wiki_documents(api_url=api_url, titles=target_titles, delay_seconds=delay_seconds)
        stats = self.upsert_documents([*house_documents, *wiki_documents])
        stats.update(
            {
                "house_policy_documents": len(house_documents),
                "wiki_documents": len(wiki_documents),
                "updated_at": _utc_now_iso(),
            }
        )
        return stats
