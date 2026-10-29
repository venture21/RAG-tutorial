"""SQLite 기반 경량 벡터 저장소 (SQLiteVectorStore)

Python 표준 라이브러리 `sqlite3`와 고속 수치 연산 라이브러리 `numpy`를 활용하여
외부 벡터 DB 서버나 C/C++ 확장 모듈 없이 파일 하나로 벡터, 원문, 메타데이터를 저장하고
코사인 유사도(Cosine Similarity) 검색을 수행합니다.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

try:
    from kiwipiepy import Kiwi
    from rank_bm25 import BM25Okapi
    HAS_HYBRID = True
except ImportError:
    Kiwi = None
    BM25Okapi = None
    HAS_HYBRID = False


class SQLiteVectorStore(VectorStore):
    """SQLite 파일 기반 경량 벡터 데이터베이스 구현체.

    LangChain의 `VectorStore` 인터페이스를 구현하며,
    각 스텝별 데이터 검증을 위한 SQLite 테이블 스키마 조회, 레코드 수 조회,
    샘플 로우 직접 확인 메서드를 제공합니다.
    """

    TABLE_NAME = "langchain_documents"

    def __init__(
        self,
        db_path: str = "rag_sqlite.db",
        embedding_function: Optional[Embeddings] = None,
    ):
        """SQLiteVectorStore 초기화.

        Args:
            db_path: SQLite DB 파일 경로 (기본값: 'rag_sqlite.db')
            embedding_function: LangChain Embeddings 객체
        """
        self.db_path = str(Path(db_path).resolve())
        self._embedding_function = embedding_function
        self._init_db()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding_function

    @contextmanager
    def _get_connection(self):
        """SQLite 데이터베이스 연결 반환."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """벡터 저장용 테이블 및 색인 초기화."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS rag_collection_config (id INTEGER PRIMARY KEY CHECK(id=1), model_name TEXT NOT NULL, dimension INTEGER NOT NULL)")
            cursor.execute("CREATE TABLE IF NOT EXISTS rag_page_manifest (source_id TEXT, page INTEGER, content_hash TEXT NOT NULL, PRIMARY KEY(source_id, page))")
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # 메타데이터의 source 및 page 필드 검색 색인 생성
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_source
                ON {self.TABLE_NAME} (json_extract(metadata, '$.source'));
                """
            )
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.TABLE_NAME}_content_hash
                ON {self.TABLE_NAME} (json_extract(metadata, '$.content_hash'));
                """
            )
            conn.commit()

    def clear(self) -> None:
        """테이블 내 모든 데이터를 삭제합니다 (재인덱싱 시 사용)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.TABLE_NAME};")
            cursor.execute("DELETE FROM rag_collection_config")
            cursor.execute("DELETE FROM rag_page_manifest")
            conn.commit()

    def collection_config(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT model_name, dimension FROM rag_collection_config WHERE id=1").fetchone()
        return dict(row) if row else None

    def assert_model(self, model_name: str, require_documents: bool = False) -> None:
        config = self.collection_config()
        count = self.get_stats()["total_records"]
        if require_documents and not count:
            raise ValueError("먼저 PDF를 업로드하고 인덱싱하세요.")
        if count and (not config or config["model_name"] != model_name):
            raise ValueError("저장된 문서와 임베딩 모델이 다릅니다. 기존 모델을 선택하거나 전체 재인덱싱하세요.")

    def commit_pages(self, documents, ids, embeddings, *, model_name, source_id, page_hashes, reindex=False):
        """Replace only covered pages, atomically, after extraction and embedding succeed."""
        if not (len(documents) == len(ids) == len(embeddings)):
            raise ValueError("청크·ID·임베딩 개수가 일치하지 않습니다.")
        vectors = [np.asarray(v, dtype=np.float32) for v in embeddings]
        dim = len(vectors[0]) if vectors else 0
        if not dim or any(v.ndim != 1 or len(v) != dim or not np.isfinite(v).all() for v in vectors):
            raise ValueError("유효하고 동일한 차원의 임베딩이 필요합니다.")
        rows = [(i, d.page_content, json.dumps(d.metadata, ensure_ascii=False), v.tobytes()) for i,d,v in zip(ids,documents,vectors)]
        with self._get_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                config = conn.execute("SELECT model_name, dimension FROM rag_collection_config WHERE id=1").fetchone()
                count = conn.execute(f"SELECT count(*) FROM {self.TABLE_NAME}").fetchone()[0]
                if not reindex and count and (not config or config["model_name"] != model_name or config["dimension"] != dim):
                    raise ValueError("다른 임베딩 모델·차원을 누적할 수 없습니다. 전체 재인덱싱하세요.")
                if reindex:
                    conn.execute(f"DELETE FROM {self.TABLE_NAME}")
                    conn.execute("DELETE FROM rag_page_manifest")
                else:
                    for page in page_hashes:
                        conn.execute(f"DELETE FROM {self.TABLE_NAME} WHERE json_extract(metadata, '$.source_id')=? AND json_extract(metadata, '$.page')=?", (source_id, page))
                conn.executemany(f"INSERT OR REPLACE INTO {self.TABLE_NAME}(id,content,metadata,embedding) VALUES (?,?,?,?)", rows)
                conn.executemany("INSERT OR REPLACE INTO rag_page_manifest(source_id,page,content_hash) VALUES (?,?,?)", [(source_id,p,h) for p,h in page_hashes.items()])
                conn.execute("INSERT OR REPLACE INTO rag_collection_config VALUES (1,?,?)", (model_name,dim))
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def delete_by_source(self, source_name: str) -> int:
        """특정 문서 소스(파일명 또는 경로)에 속한 레코드만 선택 삭제합니다."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                DELETE FROM {self.TABLE_NAME}
                WHERE json_extract(metadata, '$.source') = ?
                   OR json_extract(metadata, '$.file_name') = ?;
                """,
                (source_name, source_name),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def get_existing_content_hashes(self) -> Dict[str, str]:
        """DB에 이미 저장된 청크의 content_hash 집합을 조회합니다.
        
        Returns:
            {content_hash: source} 딕셔너리
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id,
                       json_extract(metadata, '$.content_hash') AS c_hash,
                       json_extract(metadata, '$.source') AS src,
                       json_extract(metadata, '$.file_name') AS fname
                FROM {self.TABLE_NAME};
                """
            )
            rows = cursor.fetchall()

        result: Dict[str, str] = {}
        for r in rows:
            c_hash = r["c_hash"]
            if c_hash:
                src = r["fname"] or r["src"] or r["id"]
                result[c_hash] = src
        return result

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """텍스트 목록을 임베딩하여 SQLite에 저장합니다.

        Args:
            texts: 저장할 텍스트 리스트
            metadatas: 텍스트별 메타데이터 딕셔너리 리스트
            ids: 고유 식별자 리스트 (없으면 자동 생성)
            embeddings: 이미 계산된 임베딩 벡터 목록 (선택)

        Returns:
            저장된 문서 ID 리스트
        """
        text_list = list(texts)
        if not text_list:
            return []

        # 임베딩이 미리 주어지지 않은 경우 embedding_function으로 계산
        if embeddings is None:
            if self._embedding_function is None:
                raise ValueError("임베딩 함수(embedding_function)가 지정되지 않았습니다.")
            embeddings = self._embedding_function.embed_documents(text_list)

        if ids is None:
            ids = [f"doc_{idx}" for idx in range(len(text_list))]

        if metadatas is None:
            metadatas = [{} for _ in text_list]

        if not (len(text_list) == len(ids) == len(metadatas) == len(embeddings)):
            raise ValueError("텍스트·ID·메타데이터·임베딩 개수가 일치하지 않습니다.")
        dims = {len(v) for v in embeddings}
        current_dim = self.get_stats()["vector_dimension"]
        if len(dims) != 1 or 0 in dims or (current_dim and current_dim not in dims):
            raise ValueError("동일한 차원의 벡터만 저장할 수 있습니다.")

        rows = []
        for doc_id, text, meta, emb in zip(ids, text_list, metadatas, embeddings):
            # float32 바이너리 BLOB 형태로 직렬화하여 저장
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            meta_json = json.dumps(meta, ensure_ascii=False)
            rows.append((doc_id, text, meta_json, emb_blob))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO {self.TABLE_NAME} (id, content, metadata, embedding)
                VALUES (?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()

        return ids

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """LangChain Document 객체 리스트를 저장합니다."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        kwargs_copy = dict(kwargs)
        ids = kwargs_copy.pop(
            "ids",
            [doc.metadata.get("id") or f"chunk_{i}" for i, doc in enumerate(documents)],
        )
        embeddings = kwargs_copy.pop("embeddings", None)
        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
            **kwargs_copy,
        )

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """임베딩 벡터를 기반으로 코사인 유사도 검색을 수행합니다.

        Returns:
            (Document, similarity_score) 튜플의 리스트 (점수 내림차순 정렬)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, content, metadata, embedding FROM {self.TABLE_NAME};"
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        query_vec = np.array(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            query_norm = 1e-9

        scored_results: List[Tuple[Document, float]] = []

        for row in rows:
            doc_id = row["id"]
            content = row["content"]
            meta_str = row["metadata"]
            blob = row["embedding"]

            stored_vec = np.frombuffer(blob, dtype=np.float32)
            if len(stored_vec) != len(query_vec):
                raise ValueError("질문과 저장 벡터의 차원이 다릅니다. 동일 모델로 재인덱싱하세요.")
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                stored_norm = 1e-9

            # 코사인 유사도: (A · B) / (||A|| * ||B||)
            cosine_sim = float(np.dot(query_vec, stored_vec) / (query_norm * stored_norm))

            try:
                metadata = json.loads(meta_str)
            except Exception:
                metadata = {}
            metadata["id"] = doc_id

            doc = Document(page_content=content, metadata=metadata)
            scored_results.append((doc, cosine_sim))

        # 유사도 높은 순으로 내림차순 정렬
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:k]

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 5,
        bm25_weight: float = 1.5,
        dense_weight: float = 1.0,
        rrf_k: int = 20,
        query_vector: Optional[List[float]] = None,
    ) -> List[Tuple[Document, float]]:
        """Dense 임베딩 벡터 검색과 한국어 형태소(Kiwi) BM25 키워드 검색을 RRF(Reciprocal Rank Fusion)로 융합합니다.

        특정 숫자(연도), 고유명사, 법률명 등 키워드가 정확히 일치해야 하는 질의에서
        Dense 임베딩만으로 발생하는 검색 누락을 방지하고 상위 노출을 보장합니다.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, content, metadata, embedding FROM {self.TABLE_NAME};"
            )
            rows = cursor.fetchall()

        if not rows:
            return []

        # 1. Kiwi 형태소 분석 및 BM25 랭킹 계산
        bm25_ranks: Dict[str, int] = {}
        if HAS_HYBRID and Kiwi is not None and BM25Okapi is not None:
            kiwi = Kiwi()
            # 명사, 수사, 동사/형용사 어근, 숫자 등 의미 형태소만 추출
            allowed_pos = {"NNG", "NNP", "NR", "NP", "VV", "VA", "SN", "XR"}

            def get_content_tokens(text: str) -> List[str]:
                return [t.form for t in kiwi.tokenize(text) if t.tag in allowed_pos]

            corpus = [get_content_tokens(r["content"]) for r in rows]
            q_tokens = get_content_tokens(query)
            if q_tokens:
                bm25_model = BM25Okapi(corpus)
                bm25_scores = bm25_model.get_scores(q_tokens)
                bm25_order = np.argsort(bm25_scores)[::-1]
                for rank, idx in enumerate(bm25_order, start=1):
                    bm25_ranks[rows[idx]["id"]] = rank
            else:
                for idx, r in enumerate(rows, start=1):
                    bm25_ranks[r["id"]] = idx
        else:
            for idx, r in enumerate(rows, start=1):
                bm25_ranks[r["id"]] = idx

        # 2. Dense Vector 임베딩 유사도 계산 및 랭킹
        dense_ranks: Dict[str, int] = {}
        dense_scores_map: Dict[str, float] = {}
        if query_vector is not None or self._embedding_function is not None:
            q_emb = np.array(query_vector if query_vector is not None else self._embedding_function.embed_query(query), dtype=np.float32)
            q_norm = np.linalg.norm(q_emb) or 1e-9
            dense_scores = []
            for r in rows:
                s_vec = np.frombuffer(r["embedding"], dtype=np.float32)
                if len(s_vec) != len(q_emb):
                    raise ValueError("질문과 저장 벡터의 차원이 다릅니다. 동일 모델로 재인덱싱하세요.")
                s_norm = np.linalg.norm(s_vec) or 1e-9
                sim = float(np.dot(q_emb, s_vec) / (q_norm * s_norm))
                dense_scores.append(sim)
                dense_scores_map[r["id"]] = sim
            dense_order = np.argsort(dense_scores)[::-1]
            for rank, idx in enumerate(dense_order, start=1):
                dense_ranks[rows[idx]["id"]] = rank
        else:
            for idx, r in enumerate(rows, start=1):
                dense_ranks[r["id"]] = idx
                dense_scores_map[r["id"]] = 0.5

        # 3. Reciprocal Rank Fusion (RRF) 융합
        fused_results: List[Tuple[Document, float]] = []
        for r in rows:
            doc_id = r["id"]
            r_bm25 = bm25_ranks.get(doc_id, len(rows))
            r_dense = dense_ranks.get(doc_id, len(rows))

            rrf_score = (bm25_weight / (rrf_k + r_bm25)) + (dense_weight / (rrf_k + r_dense))

            try:
                metadata = json.loads(r["metadata"])
            except Exception:
                metadata = {}
            metadata["id"] = doc_id
            metadata["bm25_rank"] = r_bm25
            metadata["dense_rank"] = r_dense
            metadata["dense_similarity"] = round(dense_scores_map.get(doc_id, 0.0), 4)

            doc = Document(page_content=r["content"], metadata=metadata)
            fused_results.append((doc, rrf_score))

        fused_results.sort(key=lambda x: x[1], reverse=True)
        return fused_results[:k]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """자연어 질문을 받아 임베딩 후 코사인 유사도 상위 k개를 반환합니다."""
        if self._embedding_function is None:
            raise ValueError("임베딩 함수가 설정되지 않았습니다.")
        query_embedding = self._embedding_function.embed_query(query)
        return self.similarity_search_by_vector_with_score(
            query_embedding, k=k, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """자연어 질문을 받아 유사한 Document 목록만 반환합니다."""
        results = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in results]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        db_path: str = "rag_sqlite.db",
        **kwargs: Any,
    ) -> "SQLiteVectorStore":
        store = cls(db_path=db_path, embedding_function=embedding)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    # =========================================================================
    # Step-by-Step 시각화 및 디버깅용 편의 메서드
    # =========================================================================
    def get_stats(self) -> Dict[str, Any]:
        """현재 SQLite 데이터베이스의 상태 통계를 조회합니다."""
        path = Path(self.db_path)
        file_size = path.stat().st_size if path.exists() else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) AS cnt FROM {self.TABLE_NAME};")
            row = cursor.fetchone()
            total_count = row["cnt"] if row else 0

            # 첫 번째 행에서 벡터 차원 수 파악
            dim = 0
            if total_count > 0:
                cursor.execute(f"SELECT embedding FROM {self.TABLE_NAME} LIMIT 1;")
                first_row = cursor.fetchone()
                if first_row and first_row["embedding"]:
                    vec = np.frombuffer(first_row["embedding"], dtype=np.float32)
                    dim = len(vec)

        return {
            "db_path": str(path),
            "table_name": self.TABLE_NAME,
            "total_records": total_count,
            "file_size_bytes": file_size,
            "vector_dimension": dim,
        }

    def get_sample_record(self) -> Optional[Dict[str, Any]]:
        """DB에 실제로 저장된 1건의 레코드를 있는 그대로 읽어옵니다."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, content, metadata, embedding, created_at
                FROM {self.TABLE_NAME}
                LIMIT 1;
                """
            )
            row = cursor.fetchone()
            if not row:
                return None

            blob = row["embedding"]
            vec = np.frombuffer(blob, dtype=np.float32)
            return {
                "id": row["id"],
                "content_preview": (row["content"][:100] + "...")
                if len(row["content"]) > 100
                else row["content"],
                "content_length": len(row["content"]),
                "metadata": json.loads(row["metadata"]),
                "embedding_blob_bytes": len(blob),
                "embedding_dimension": len(vec),
                "embedding_preview": [round(float(v), 4) for v in vec[:5]],
                "created_at": row["created_at"],
            }
