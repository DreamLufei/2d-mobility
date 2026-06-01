from __future__ import annotations

import hashlib
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore


_MEMORY_LOCK = Lock()
_MEMORY_CHECKPOINTERS: dict[str, InMemorySaver] = {}
_MEMORY_STORES: dict[str, InMemoryStore] = {}


def is_postgres_uri(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return text.startswith("postgresql://") or text.startswith("postgres://")


def is_memory_uri(value: str | None) -> bool:
    return str(value or "").strip().lower().startswith("memory://")


def normalize_database_uri(value: str | None, *, default_memory_name: str = "default") -> str:
    text = str(value or "").strip()
    if is_postgres_uri(text) or is_memory_uri(text):
        return text
    if not text:
        return f"memory://{default_memory_name}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"memory://{digest}"


def redact_database_uri(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if is_memory_uri(text):
        return text
    if "@" not in text:
        return text
    prefix, suffix = text.split("@", 1)
    if "://" not in prefix:
        return text
    scheme, auth = prefix.split("://", 1)
    if ":" in auth:
        username = auth.split(":", 1)[0]
        return f"{scheme}://{username}:***@{suffix}"
    return f"{scheme}://***@{suffix}"


def sqlalchemy_database_uri(value: str) -> str:
    text = normalize_database_uri(value)
    if text.startswith("postgresql+psycopg://"):
        return text
    if text.startswith("postgresql://"):
        return text.replace("postgresql://", "postgresql+psycopg://", 1)
    if text.startswith("postgres://"):
        return text.replace("postgres://", "postgresql+psycopg://", 1)
    return text


@contextmanager
def open_checkpointer(database_uri: str) -> Iterator[BaseCheckpointSaver]:
    resolved = normalize_database_uri(database_uri)
    if is_memory_uri(resolved):
        with _MEMORY_LOCK:
            saver = _MEMORY_CHECKPOINTERS.get(resolved)
            if saver is None:
                saver = InMemorySaver()
                _MEMORY_CHECKPOINTERS[resolved] = saver
        yield saver
        return
    with PostgresSaver.from_conn_string(resolved) as saver:
        saver.setup()
        yield saver


@contextmanager
def open_kv_store(database_uri: str) -> Iterator[BaseStore]:
    resolved = normalize_database_uri(database_uri)
    if is_memory_uri(resolved):
        with _MEMORY_LOCK:
            store = _MEMORY_STORES.get(resolved)
            if store is None:
                store = InMemoryStore()
                _MEMORY_STORES[resolved] = store
        yield store
        return
    with PostgresStore.from_conn_string(resolved) as store:
        store.setup()
        yield store


def checkpoint_exists(*, database_uri: str, thread_id: str | None) -> bool:
    resolved_thread_id = str(thread_id or "").strip()
    if not resolved_thread_id:
        return False
    with open_checkpointer(database_uri) as saver:
        return saver.get_tuple({"configurable": {"thread_id": resolved_thread_id}}) is not None


def delete_checkpoint_thread(*, database_uri: str, thread_id: str | None) -> None:
    resolved_thread_id = str(thread_id or "").strip()
    if not resolved_thread_id:
        return
    with open_checkpointer(database_uri) as saver:
        try:
            saver.delete_thread(resolved_thread_id)
        except Exception:
            return
