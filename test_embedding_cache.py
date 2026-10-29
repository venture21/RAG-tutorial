"""임베딩 캐시 (EmbeddingCache & CachedEmbeddings & API) 단위 및 통합 테스트."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app_rag import create_app
from embedding_cache import CachedEmbeddings, EmbeddingCache
from rag_service import RAGService


class MockEmbedder(Embeddings):
    """테스트용 가상 임베더 (호출 횟수 계측)."""

    def __init__(self, model_name: str = "text-embedding-004"):
        self.model = model_name
        self.client = "mock-client"
        self.doc_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts):
        self.doc_calls += len(texts)
        return [[float(len(t)), float(idx)] for idx, t in enumerate(texts)]

    def embed_query(self, text):
        self.query_calls += 1
        return [float(len(text)), 1.234]


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.cache = EmbeddingCache(cache_dir=self.cache_dir, db_name="test_emb.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_text_normalization_and_key_consistency(self):
        t1 = "  안녕하세요   반갑습니다 \n\n 오늘 날씨가 좋네요  "
        t2 = "안녕하세요 반갑습니다 오늘 날씨가 좋네요"
        key1 = self.cache.compute_cache_key("text-embedding-004", t1)
        key2 = self.cache.compute_cache_key("text-embedding-004", t2)
        self.assertEqual(key1, key2, "공백/줄바꿈 차이가 있더라도 정규화되어 동일한 캐시 키를 생성해야 합니다.")

    def test_get_and_set_single(self):
        model = "text-embedding-004"
        text = "테스트 문장"
        vec = [0.1, 0.2, 0.3]

        self.assertIsNone(self.cache.get(text, model))
        self.cache.set(text, model, vec)

        cached_vec = self.cache.get(text, model)
        self.assertIsNotNone(cached_vec)
        self.assertEqual(len(cached_vec), 3)
        self.assertAlmostEqual(cached_vec[0], 0.1, places=5)

    def test_get_many_and_set_many(self):
        model = "text-embedding-004"
        texts = ["문장 1", "문장 2", "문장 3"]
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        # 처음에는 모두 누락
        cached_dict, missing_items = self.cache.get_many(texts, model)
        self.assertEqual(len(cached_dict), 0)
        self.assertEqual(len(missing_items), 3)

        # 일괄 저장
        self.cache.set_many(texts, model, vectors)

        # 재조회 시 모두 적중
        cached_dict2, missing_items2 = self.cache.get_many(texts, model)
        self.assertEqual(len(cached_dict2), 3)
        self.assertEqual(len(missing_items2), 0)
        self.assertEqual(cached_dict2[0], vectors[0])
        self.assertEqual(cached_dict2[1], vectors[1])
        self.assertEqual(cached_dict2[2], vectors[2])

    def test_cached_embeddings_batch_workflow(self):
        raw_embedder = MockEmbedder(model_name="test-model")
        cached_embedder = CachedEmbeddings(raw_embedder, cache=self.cache)

        # 1. 1차 호출: 3건 모두 원격 호출
        texts = ["청크 A", "청크 B", "청크 C"]
        res1 = cached_embedder.embed_documents(texts)
        self.assertEqual(len(res1), 3)
        self.assertEqual(raw_embedder.doc_calls, 3)
        self.assertEqual(cached_embedder.last_batch_stats["hits"], 0)
        self.assertEqual(cached_embedder.last_batch_stats["misses"], 3)
        self.assertFalse(cached_embedder.last_batch_stats["from_cache"])

        # 2. 2차 호출 (완전 동일): API 호출 0건, 캐시 적중 100%
        res2 = cached_embedder.embed_documents(texts)
        self.assertEqual(res1, res2)
        self.assertEqual(raw_embedder.doc_calls, 3, "동일 텍스트 호출 시 원격 API를 추가 호출하지 않아야 합니다.")
        self.assertEqual(cached_embedder.last_batch_stats["hits"], 3)
        self.assertEqual(cached_embedder.last_batch_stats["misses"], 0)
        self.assertTrue(cached_embedder.last_batch_stats["from_cache"])

        # 3. 3차 호출 (부분 중복): 청크 B, 청크 신규, 청크 A
        mixed = ["청크 B", "새로운 청크 D", "청크 A"]
        res3 = cached_embedder.embed_documents(mixed)
        self.assertEqual(len(res3), 3)
        self.assertEqual(raw_embedder.doc_calls, 4, "누락된 1건에 대해서만 API를 호출해야 합니다.")
        self.assertEqual(cached_embedder.last_batch_stats["hits"], 2)
        self.assertEqual(cached_embedder.last_batch_stats["misses"], 1)

    def test_cached_embeddings_query_workflow(self):
        raw_embedder = MockEmbedder(model_name="query-model")
        cached_embedder = CachedEmbeddings(raw_embedder, cache=self.cache)

        q = "질문입니다"
        q_vec1 = cached_embedder.embed_query(q)
        self.assertEqual(raw_embedder.query_calls, 1)
        self.assertFalse(cached_embedder.last_batch_stats["from_cache"])

        # 동일 질문 재호출
        q_vec2 = cached_embedder.embed_query(q)
        import numpy as np
        np.testing.assert_allclose(q_vec1, q_vec2, atol=1e-5)
        self.assertEqual(raw_embedder.query_calls, 1, "동일 질문은 캐시에서 즉각 반환되어야 합니다.")
        self.assertTrue(cached_embedder.last_batch_stats["from_cache"])

    def test_cache_stats_and_clear(self):
        raw_embedder = MockEmbedder(model_name="stat-model")
        cached_embedder = CachedEmbeddings(raw_embedder, cache=self.cache)

        cached_embedder.embed_documents(["가", "나", "다"])
        cached_embedder.embed_documents(["가", "나"])

        stats = self.cache.get_stats()
        self.assertEqual(stats["total_cached_entries"], 3)
        self.assertGreaterEqual(stats["lifetime_hits"], 2)
        self.assertGreater(stats["file_size_bytes"], 0)

        # 모델별 삭제
        deleted = self.cache.clear(model_name="stat-model")
        self.assertEqual(deleted, 3)
        stats2 = self.cache.get_stats()
        self.assertEqual(stats2["total_cached_entries"], 0)


class FastAPICacheEndpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        service = RAGService(data_dir=directory.name)
        self.client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")

    def test_cache_stats_endpoint(self):
        res = self.client.get("/api/cache/embedding/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_cached_entries", data)
        self.assertIn("session_hits", data)
        self.assertIn("file_size_kb", data)

    def test_cache_clear_endpoint(self):
        res = self.client.post("/api/cache/embedding/clear")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("deleted_entries", data)


if __name__ == "__main__":
    unittest.main()
