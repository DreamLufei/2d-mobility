#!/usr/bin/env python3
from __future__ import annotations

import os

from mobility_agent.rag import VaspWikiRagService


def main() -> int:
    runtime_db_uri = os.environ.get("MOBILITY_DB_URI") or ""
    embedding_model = os.environ.get("EMBEDDING_MODEL") or ""
    if not runtime_db_uri:
        raise RuntimeError("MOBILITY_DB_URI is required.")
    if not embedding_model:
        raise RuntimeError("EMBEDDING_MODEL is required.")
    service = VaspWikiRagService(
        database_uri=runtime_db_uri,
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
    )
    print(service.rebuild_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
