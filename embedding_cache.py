"""SQLite 기반 임베딩 캐시 (EmbeddingCache & CachedEmbeddings)

임베딩 모델(Gemini text-embedding-004 등)의 API 호출 결과를 로컬 SQLite 데이터베이스에 캐싱하여,
동일한 텍스트 청크나 질문에 대해 불필요한 원격 API 호출을 방지하고
비용 절감, 속도 향상(수 초 -> 0초), Quota 429 에러 방지를 제공합니다.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings


class EmbeddingCache:
    """SQLite 파일 기반 임베딩 벡터 캐시 저장소.
    
    각 임베딩 벡터를 float32 바이너리(BLOB)로 압축 저장하며,
    모델명과 텍스트 내용을 조합한 SHA-256 해시를 키로 사용합니다.
    """

    TABLE_NAME = "embedding_cache"

    def __init__(self, cache_dir: Optional[str | Path] = None, db_name: str = "embeddings.db"):
        if cache_dir:
            base_dir = Path(cache_dir).resolve()
        else:
            # 기본 경로: RAG-tutorial/.cache/
            base_dir = Path(__file__).resolve().parent / ".cache"
        
        base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str((base_dir / db_name).resolve())

        # 런타임 캐시 적중/미스 통계
        self._session_hits = 0
        self._session_misses = 0

        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """캐시 테이블 및 인덱스 초기화 (WAL 모드 적용)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 동시성 및 쓰기 성능 향상을 위한 WAL 모드 활성화
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    text_preview TEXT NOT NULL,
                    char_length INTEGER NOT NULL,
                    dimension INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                );
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_model
                ON {self.TABLE_NAME} (model_name);
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_text_hash
                ON {self.TABLE_NAME} (text_hash);
                """
            )
            conn.commit()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """불필요한 공백을 정규화하여 공백 차이로 인한 캐시 미스를 방지합니다."""
        return " ".join(text.split())

    @classmethod
    def compute_text_hash(cls, text: str) -> str:
        """정규화된 텍스트의 SHA-256 해시를 계산합니다."""
        norm = cls._normalize_text(text)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    @classmethod
    def compute_cache_key(cls, model_name: str, text: str) -> str:
        """모델명과 텍스트 내용을 조합한 고유 캐시 키를 생성합니다."""
        norm = cls._normalize_text(text)
        payload = f"{model_name.strip()}::{norm}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """단일 텍스트에 대한 캐시된 임베딩을 조회합니다."""
        key = self.compute_cache_key(model_name, text)
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT embedding, dimension
                FROM {self.TABLE_NAME}
                WHERE cache_key = ?;
                """,
                (key,),
            )
            row = cursor.fetchone()
            if row:
                cursor.execute(
                    f"""
                    UPDATE {self.TABLE_NAME}
                    SET hit_count = hit_count + 1,
                        last_accessed_at = ?
                    WHERE cache_key = ?;
                    """,
                    (now, key),
                )
                conn.commit()
                self._session_hits += 1
                return np.frombuffer(row["embedding"], dtype=np.float32).tolist()

        self._session_misses += 1
        return None

    def get_many(
        self, texts: Sequence[str], model_name: str
    ) -> Tuple[Dict[int, List[float]], List[Tuple[int, str]]]:
        """텍스트 목록을 일괄 조회하여 캐시 적중 항목과 누락(미스) 항목으로 분리합니다.

        Returns:
            (cached_dict, missing_list)
            - cached_dict: {원래 인덱스: 임베딩 벡터}
            - missing_list: [(원래 인덱스, 텍스트), ...]
        """
        if not texts:
            return {}, []

        keys = [self.compute_cache_key(model_name, t) for t in texts]
        cached_results: Dict[int, List[float]] = {}
        missing_items: List[Tuple[int, str]] = []

        # SQLite IN 절을 위한 키 매핑 (동일 텍스트가 여러 인덱스에 존재할 수 있음)
        key_to_indices: Dict[str, List[int]] = {}
        for idx, k in enumerate(keys):
            key_to_indices.setdefault(k, []).append(idx)

        unique_keys = list(key_to_indices.keys())
        hit_keys: set[str] = set()

        batch_size = 900  # SQLite 파라미터 한도 안전값
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for b_start in range(0, len(unique_keys), batch_size):
                chunk = unique_keys[b_start : b_start + batch_size]
                placeholders = ",".join(["?"] * len(chunk))
                cursor.execute(
                    f"""
                    SELECT cache_key, embedding
                    FROM {self.TABLE_NAME}
                    WHERE cache_key IN ({placeholders});
                    """,
                    chunk,
                )
                rows = cursor.fetchall()
                for r in rows:
                    k = r["cache_key"]
                    hit_keys.add(k)
                    vec = np.frombuffer(r["embedding"], dtype=np.float32).tolist()
                    for idx in key_to_indices[k]:
                        cached_results[idx] = vec

            # 적중된 항목들의 hit_count 및 last_accessed_at 일괄 갱신
            if hit_keys:
                hit_key_list = list(hit_keys)
                for b_start in range(0, len(hit_key_list), batch_size):
                    chunk = hit_key_list[b_start : b_start + batch_size]
                    placeholders = ",".join(["?"] * len(chunk))
                    cursor.execute(
                        f"""
                        UPDATE {self.TABLE_NAME}
                        SET hit_count = hit_count + 1,
                            last_accessed_at = ?
                        WHERE cache_key IN ({placeholders});
                        """,
                        [now] + chunk,
                    )
                conn.commit()

        # 미적중 항목 정리
        for idx, text in enumerate(texts):
            if idx not in cached_results:
                missing_items.append((idx, text))

        hits = len(cached_results)
        misses = len(missing_items)
        self._session_hits += hits
        self._session_misses += misses

        return cached_results, missing_items

    def set(self, text: str, model_name: str, embedding: Sequence[float]) -> None:
        """단일 텍스트 임베딩을 캐시에 저장합니다."""
        key = self.compute_cache_key(model_name, text)
        t_hash = self.compute_text_hash(text)
        preview = self._normalize_text(text)[:120]
        dim = len(embedding)
        blob = np.array(embedding, dtype=np.float32).tobytes()
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                INSERT INTO {self.TABLE_NAME}
                    (cache_key, model_name, text_hash, text_preview, char_length, dimension, embedding, created_at, last_accessed_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    embedding = excluded.embedding,
                    dimension = excluded.dimension,
                    last_accessed_at = excluded.last_accessed_at;
                """,
                (key, model_name.strip(), t_hash, preview, len(text), dim, blob, now, now),
            )
            conn.commit()

    def set_many(
        self, texts: Sequence[str], model_name: str, embeddings: Sequence[Sequence[float]]
    ) -> None:
        """여러 개의 텍스트 임베딩을 일괄 저장합니다."""
        if not texts or not embeddings or len(texts) != len(embeddings):
            return

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        m_name = model_name.strip()
        rows = []

        for text, emb in zip(texts, embeddings):
            key = self.compute_cache_key(m_name, text)
            t_hash = self.compute_text_hash(text)
            preview = self._normalize_text(text)[:120]
            dim = len(emb)
            blob = np.array(emb, dtype=np.float32).tobytes()
            rows.append((key, m_name, t_hash, preview, len(text), dim, blob, now, now))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                f"""
                INSERT INTO {self.TABLE_NAME}
                    (cache_key, model_name, text_hash, text_preview, char_length, dimension, embedding, created_at, last_accessed_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(cache_key) DO UPDATE SET
                    embedding = excluded.embedding,
                    dimension = excluded.dimension,
                    last_accessed_at = excluded.last_accessed_at;
                """,
                rows,
            )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """캐시 현황 및 세션 적중 통계를 반환합니다."""
        db_file = Path(self.db_path)
        file_size = db_file.stat().st_size if db_file.exists() else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) AS total FROM {self.TABLE_NAME};")
            total_records = cursor.fetchone()["total"]

            cursor.execute(
                f"""
                SELECT model_name, COUNT(*) AS cnt, SUM(hit_count) AS hits
                FROM {self.TABLE_NAME}
                GROUP BY model_name;
                """
            )
            model_stats = {
                r["model_name"]: {"entries": r["cnt"], "total_hits": r["hits"] or 0}
                for r in cursor.fetchall()
            }

            cursor.execute(
                f"SELECT SUM(hit_count) AS total_lifetime_hits FROM {self.TABLE_NAME};"
            )
            lifetime_hits = cursor.fetchone()["total_lifetime_hits"] or 0

        total_requests = self._session_hits + self._session_misses
        session_hit_rate = (
            round((self._session_hits / total_requests * 100), 1)
            if total_requests > 0
            else 0.0
        )

        return {
            "db_path": self.db_path,
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 1),
            "total_cached_entries": total_records,
            "lifetime_hits": lifetime_hits,
            "models": model_stats,
            "session_hits": self._session_hits,
            "session_misses": self._session_misses,
            "session_hit_rate": session_hit_rate,
        }

    def clear(self, model_name: Optional[str] = None) -> int:
        """캐시를 초기화합니다. 모델명이 지정되면 해당 모델의 캐시만 삭제합니다."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if model_name:
                prefix = model_name.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "::cache-v2::%"
                cursor.execute(
                    f"DELETE FROM {self.TABLE_NAME} WHERE model_name = ? OR model_name LIKE ? ESCAPE '\\';",
                    (model_name.strip(), prefix),
                )
            else:
                cursor.execute(f"DELETE FROM {self.TABLE_NAME};")
            deleted = cursor.rowcount
            conn.commit()
            if not model_name:
                cursor.execute("VACUUM;")
                conn.commit()
        return deleted


class CachedEmbeddings(Embeddings):
    """LangChain `Embeddings` 인터페이스를 구현한 캐싱 래퍼.
    
    기존 GoogleGenerativeAIEmbeddings 인스턴스를 감싸서
    문서 인덱싱(embed_documents) 및 질의(embed_query) 시
    캐시 확인 -> 미스 항목만 원격 모델 호출 -> 캐시 저장 흐름을 투명하게 처리합니다.
    """

    def __init__(
        self,
        underlying_embeddings: Embeddings,
        cache: Optional[EmbeddingCache] = None,
        model_name: Optional[str] = None,
    ):
        self.underlying_embeddings = underlying_embeddings
        self.cache = cache or EmbeddingCache()
        
        # 모델명 추출: 전달값 -> underlying_embeddings.model -> 기본값
        self.model_name = (
            model_name
            or getattr(underlying_embeddings, "model", None)
            or "default-embedding"
        )
        self.last_batch_stats = {"total": 0, "hits": 0, "misses": 0, "hit_rate": 0.0, "from_cache": False}

    def cache_model_name(self, purpose: str) -> str:
        """Namespace by task and output configuration, never by credentials."""
        task = getattr(self.underlying_embeddings, "task_type", None)
        task = task if isinstance(task, str) else ("RETRIEVAL_QUERY" if purpose == "query" else "RETRIEVAL_DOCUMENT")
        dim = getattr(self.underlying_embeddings, "output_dimensionality", None)
        dim = dim if isinstance(dim, int) else None
        return self.model_name + "::cache-v2::" + json.dumps({"task": task, "dimension": dim}, sort_keys=True)

    def __getattr__(self, name: str) -> Any:
        """underlying_embeddings의 속성(예: client, project 등)을 투명하게 위임합니다."""
        return getattr(self.underlying_embeddings, name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서 청크의 임베딩을 계산합니다.
        
        캐시에 이미 존재하는 청크는 즉각 로컬에서 반환하고,
        누락된 청크만 underlying_embeddings.embed_documents()로 일괄 계산하여 저장합니다.
        """
        if not texts:
            self.last_batch_stats = {
                "total": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "from_cache": True,
            }
            return []

        namespace = self.cache_model_name("document")
        cached_dict, missing_items = self.cache.get_many(texts, namespace)
        total = len(texts)
        hits = len(cached_dict)
        misses = len(missing_items)

        if missing_items:
            # 캐시에 없는 텍스트만 원격 API 호출
            missing_texts = [text for _, text in missing_items]
            computed_embeddings = self.underlying_embeddings.embed_documents(missing_texts)

            # 새로 계산된 임베딩 캐시 저장
            self.cache.set_many(missing_texts, namespace, computed_embeddings)

            # 원래 인덱스 매핑
            for (idx, _), vec in zip(missing_items, computed_embeddings):
                cached_dict[idx] = vec

        hit_rate = round((hits / total * 100), 1) if total > 0 else 0.0
        self.last_batch_stats = {
            "total": total,
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "from_cache": (misses == 0 and hits > 0),
        }

        # 원래 텍스트 순서대로 임베딩 리스트 정렬하여 반환
        return [cached_dict[i] for i in range(total)]

    def embed_query(self, text: str) -> List[float]:
        """단일 질문 텍스트에 대한 임베딩을 계산합니다."""
        namespace = self.cache_model_name("query")
        cached_vec = self.cache.get(text, namespace)
        if cached_vec is not None:
            self.last_batch_stats = {
                "total": 1,
                "hits": 1,
                "misses": 0,
                "hit_rate": 100.0,
                "from_cache": True,
            }
            return cached_vec

        computed = self.underlying_embeddings.embed_query(text)
        self.cache.set(text, namespace, computed)
        self.last_batch_stats = {
            "total": 1,
            "hits": 0,
            "misses": 1,
            "hit_rate": 0.0,
            "from_cache": False,
        }
        return computed

