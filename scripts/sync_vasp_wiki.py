#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from mobility_agent.rag import VaspWikiRagService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync VASP Wiki and house policy documents into Postgres.")
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    parser.add_argument("--all-pages", action="store_true", help="Fetch all pages from the VASP Wiki.")
    parser.add_argument("--max-pages", type=int, default=0, help="Maximum number of wiki pages to fetch. 0 means default set.")
    parser.add_argument("--delay-seconds", type=float, default=0.2, help="Delay between wiki fetches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    payload = service.sync(
        mode=args.mode,
        include_all_pages=bool(args.all_pages),
        max_pages=(args.max_pages if args.max_pages > 0 else None),
        delay_seconds=max(0.0, float(args.delay_seconds or 0.0)),
    )
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
