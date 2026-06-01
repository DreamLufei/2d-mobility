from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class WikiDocument(BaseModel):
    corpus: str
    source_id: str
    revision_id: str = ""
    content_sha: str
    title: str
    heading: str = ""
    stage: str = ""
    tags: list[str] = Field(default_factory=list)
    url: str = ""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self):
        self.corpus = str(self.corpus or "unknown")
        self.source_id = str(self.source_id or "unknown")
        self.revision_id = str(self.revision_id or "")
        self.content_sha = str(self.content_sha or "")
        self.title = str(self.title or self.source_id)
        self.heading = str(self.heading or self.title or self.source_id)
        self.stage = str(self.stage or "")
        self.tags = [str(item) for item in list(self.tags or []) if str(item)]
        self.url = str(self.url or "")
        self.text = str(self.text or "")
        self.metadata = dict(self.metadata or {})
        return self


class WikiChunk(BaseModel):
    chunk_id: str
    corpus: str
    source_id: str
    revision_id: str = ""
    content_sha: str
    title: str
    heading: str = ""
    stage: str = ""
    tags: list[str] = Field(default_factory=list)
    url: str = ""
    text: str
    sequence: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagCitation(BaseModel):
    corpus: str
    source_id: str
    chunk_id: str
    revision_id: str = ""
    title: str
    heading: str = ""
    url: str = ""
    snippet: str
    score: float = 0.0
    stage: str = ""
    tags: list[str] = Field(default_factory=list)


class RagAnswer(BaseModel):
    query: str
    answer: str
    citations: list[RagCitation] = Field(default_factory=list)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 6
    corpora: list[str] = Field(default_factory=list)
    stage: str = ""


class RagQueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[RagCitation] = Field(default_factory=list)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
