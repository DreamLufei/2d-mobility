from __future__ import annotations

import unittest

from unittest.mock import patch

from mobility_agent.rag.service import OpenRouterEmbeddings, VaspWikiRagService


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return dict(self._payload)


class OpenRouterEmbeddingsTests(unittest.TestCase):
    def test_embed_query_reads_standard_openrouter_embedding_payload(self) -> None:
        embeddings = OpenRouterEmbeddings(
            model="thenlper/gte-base",
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(
            "mobility_agent.rag.service.requests.post",
            return_value=_FakeResponse({"object": "list", "data": [{"embedding": [0.1, 0.2, 0.3]}]}),
        ):
            vector = embeddings.embed_query("hello")

        self.assertEqual(vector, [0.1, 0.2, 0.3])

    def test_embed_query_raises_router_error_message(self) -> None:
        embeddings = OpenRouterEmbeddings(
            model="openai/text-embedding-3-large",
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
        )
        with patch(
            "mobility_agent.rag.service.requests.post",
            return_value=_FakeResponse({"error": {"message": "No successful provider responses.", "code": 404}}),
        ):
            with self.assertRaisesRegex(ValueError, "No successful provider responses"):
                embeddings.embed_query("hello")

    def test_service_uses_openrouter_embedding_adapter(self) -> None:
        service = VaspWikiRagService(
            database_uri="postgresql://example",
            embedding_model="perplexity/pplx-embed-v1-4b",
            embedding_base_url="https://openrouter.ai/api/v1",
            embedding_api_key="sk-test",
            qa_model="dummy",
            qa_base_url="https://example.invalid/v1",
            qa_api_key="sk-test",
        )

        self.assertIsInstance(service._embeddings(), OpenRouterEmbeddings)

    def test_service_uses_openai_embeddings_for_dashscope(self) -> None:
        service = VaspWikiRagService(
            database_uri="postgresql://example",
            embedding_model="text-embedding-v4",
            embedding_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            embedding_api_key="sk-test",
            qa_model="qwen3.6-plus",
            qa_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            qa_api_key="sk-test",
        )

        with patch("mobility_agent.rag.service.OpenAIEmbeddings") as mock_embeddings:
            embeddings = service._embeddings()

        self.assertIs(embeddings, mock_embeddings.return_value)
        _, kwargs = mock_embeddings.call_args
        self.assertEqual(kwargs["model"], "text-embedding-v4")
        self.assertEqual(kwargs["base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")


if __name__ == "__main__":
    unittest.main()
