from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from .models import WikiDocument


DEFAULT_API_URL = "https://www.vasp.at/wiki/api.php"
DEFAULT_PAGE_TITLES = [
    "The_VASP_Manual",
    "INCAR",
    "KPOINTS",
    "POSCAR",
    "POTCAR",
    "ISMEAR",
    "SIGMA",
    "ENCUT",
    "EDIFF",
    "EDIFFG",
    "IBRION",
    "ISIF",
    "NSW",
    "ALGO",
    "PREC",
    "LASPH",
    "LREAL",
    "ISYM",
    "NCORE",
    "KPAR",
    "Band-structure_calculation",
]


def _json_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode(params, doseq=True)
    with urlopen(f"{url}?{query}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_title(title: str) -> str:
    return str(title or "").strip().replace(" ", "_")


def infer_tags(title: str, text: str) -> list[str]:
    haystack = f"{title}\n{text}".lower()
    tags: list[str] = []
    keyword_map = {
        "relax": ["relax", "ibrion", "isif", "nsw", "ediffg", "ionic"],
        "scf": ["scf", "electronic minimization", "self-consistent", "ismear", "sigma"],
        "band": ["band", "line mode", "high-symmetry", "band structure"],
        "kpoints": ["kpoints", "monkhorst", "gamma", "line mode"],
        "parallel": ["ncore", "kpar", "npar", "mpi", "parallel"],
    }
    for tag, keywords in keyword_map.items():
        if any(keyword in haystack for keyword in keywords):
            tags.append(tag)
    return tags


def infer_stage(title: str, text: str) -> str:
    tags = infer_tags(title, text)
    for stage in ("relax", "scf", "band"):
        if stage in tags:
            return stage
    return ""


def clean_wikitext(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"<ref[^>]*>.*?</ref>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    for _ in range(4):
        updated = re.sub(r"\{\{[^{}|]+\|([^{}]+?)\}\}", r"\1", value)
        updated = re.sub(r"\{\{([^{}|]+?)\}\}", r"\1", updated)
        if updated == value:
            break
        value = updated
    value = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", value)
    value = re.sub(r"\[https?://[^\s\]]+\]", " ", value)
    value = re.sub(r"'''?", "", value)
    value = re.sub(r"__\w+__", " ", value)
    value = re.sub(r"\[\[:?Category:[^\]]+\]\]", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _content_sha(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()


def fetch_wiki_document(api_url: str, title: str) -> WikiDocument | None:
    payload = _json_get(
        api_url,
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "revisions|info",
            "titles": title,
            "rvprop": "ids|content",
            "rvslots": "main",
            "redirects": 1,
            "inprop": "url",
        },
    )
    pages = ((payload.get("query") or {}).get("pages") or [])
    for item in pages:
        if not isinstance(item, dict):
            continue
        page_title = str(item.get("title") or title)
        revisions = list(item.get("revisions") or [])
        revision = revisions[0] if revisions else {}
        slots = dict((revision or {}).get("slots") or {})
        main_slot = dict(slots.get("main") or {})
        raw_text = str(main_slot.get("content") or "").strip()
        cleaned_text = clean_wikitext(raw_text)
        if not cleaned_text:
            return None
        return WikiDocument(
            corpus="vasp_wiki",
            source_id=page_title,
            revision_id=str(revision.get("revid") or ""),
            content_sha=_content_sha(cleaned_text),
            title=page_title,
            heading=page_title,
            stage=infer_stage(page_title, cleaned_text),
            tags=infer_tags(page_title, cleaned_text),
            url=str(item.get("fullurl") or ""),
            text=cleaned_text,
            metadata={"pageid": item.get("pageid"), "title": page_title},
        )
    return None


def iter_all_page_titles(api_url: str, *, max_pages: int | None = None) -> list[str]:
    titles: list[str] = []
    continuation: dict[str, Any] = {}
    while True:
        payload = _json_get(
            api_url,
            {
                "action": "query",
                "format": "json",
                "list": "allpages",
                "apnamespace": 0,
                "aplimit": "max",
                **continuation,
            },
        )
        batch = ((payload.get("query") or {}).get("allpages") or [])
        for item in batch:
            if not isinstance(item, dict):
                continue
            titles.append(normalize_title(item.get("title") or ""))
            if max_pages is not None and len(titles) >= max_pages:
                return titles
        continuation = payload.get("continue") or {}
        if not continuation:
            return titles


def sync_wiki_documents(
    *,
    api_url: str,
    titles: list[str],
    delay_seconds: float,
) -> list[WikiDocument]:
    documents: list[WikiDocument] = []
    seen: set[str] = set()
    for title in titles:
        normalized = normalize_title(title)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        document = fetch_wiki_document(api_url, normalized)
        if document is not None:
            documents.append(document)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    return documents


def load_house_policy_documents(path: str) -> list[WikiDocument]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_documents = payload.get("documents", payload) if isinstance(payload, dict) else payload
    documents: list[WikiDocument] = []
    for item in list(raw_documents or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        documents.append(
            WikiDocument(
                corpus=str(item.get("corpus") or "house_policy"),
                source_id=str(item.get("source_id") or item.get("title") or "house-policy"),
                revision_id=str(item.get("revision_id") or ""),
                content_sha=_content_sha(text),
                title=str(item.get("title") or item.get("source_id") or "house-policy"),
                heading=str(item.get("heading") or item.get("title") or ""),
                stage=str(item.get("stage") or ""),
                tags=[str(tag) for tag in list(item.get("tags") or []) if str(tag)],
                url=str(item.get("url_or_path") or ""),
                text=text,
                metadata={"source": "house_policy_json"},
            )
        )
    return documents
