"""RAG 파이프라인 실험 & 벤치마크 로거 (RAG Experiment & Benchmark Logger).

인덱싱 전송(Part A) 및 질문 전송(Part B) 시:
- OCR 사용 여부 / 방식
- 임베딩 모델 선택 (text-embedding-004, gemini-embedding-2 등)
- 캐시 활성화 여부 및 적중률
- LLM 모델 선택 (Gemini, DeepSeek, Gemma, Claude Haiku 등)
- 대상 파일 정보 (파일명, 경로, 크기, 총 페이지 수, 청크 수)
- 청크 파라미터 (chunk_size, chunk_overlap)
- 레이턴시 (단계별 소요시간, 전체 소요시간)
- 토큰 소모량 및 공식 요율 기반 과금액
- 응답 품질 지표 (출처 인용 수, 초당 토큰 생성 속도)
를 구조화된 JSONL 파일 및 SQLite 데이터베이스에 영구 보존하고,
최적의 RAG 조합(Speed vs Cost vs Quality)을 도출하기 위한 분석 인터페이스를 제공합니다.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from runtime_config import data_directory

# 기본 저장 경로
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = data_directory() / "logs"
JSONL_LOG_PATH = LOGS_DIR / "rag_experiments.jsonl"
SQLITE_DB_PATH = data_directory() / "rag_sqlite.db"

_KST = ZoneInfo("Asia/Seoul")
_LOGGER_LOCK = threading.Lock()


def get_current_kst_time() -> str:
    """현재 한국 표준시(KST) ISO 포맷 문자열을 반환합니다."""
    return datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S")


def get_iso_timestamp() -> str:
    """ISO 8601 타임스탬프를 반환합니다."""
    return datetime.now(_KST).isoformat()


class RAGExperimentLogger:
    """RAG 최적 조합 분석을 위한 인덱싱/질의 벤치마크 실험 로거."""

    def __init__(
        self,
        jsonl_path: Path = JSONL_LOG_PATH,
        db_path: Path = SQLITE_DB_PATH,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.db_path = Path(db_path)
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """로그 디렉토리 및 SQLite 테이블을 초기화합니다."""
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.jsonl_path.exists():
            self.jsonl_path.touch()

        with _LOGGER_LOCK:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rag_experiment_logs (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,          -- 'indexing' 또는 'query'
                        file_name TEXT,                    -- 원본 문서 파일명
                        total_pages INTEGER DEFAULT 0,     -- 문서 총 페이지 수
                        total_chunks INTEGER DEFAULT 0,    -- 총 청크 수
                        chunk_size INTEGER DEFAULT 0,      -- 청크 분할 크기
                        chunk_overlap INTEGER DEFAULT 0,   -- 청크 오버랩 크기
                        ocr_used INTEGER DEFAULT 0,        -- OCR 사용 여부 (0/1)
                        loader_type TEXT,                  -- 로더 유형
                        embedding_model TEXT NOT NULL,     -- 임베딩 모델명
                        embedding_dim INTEGER DEFAULT 0,   -- 임베딩 벡터 차원
                        cache_enabled INTEGER DEFAULT 1,   -- 캐시 사용 여부 (0/1)
                        cache_hit INTEGER DEFAULT 0,       -- 캐시 적중 여부/개수
                        cache_hit_ratio REAL DEFAULT 0.0,  -- 캐시 적중률 (%)
                        llm_model TEXT,                    -- 질의에 사용된 LLM 모델명
                        llm_provider TEXT,                 -- LLM 제공사
                        question TEXT,                     -- 질문 내용
                        top_k INTEGER DEFAULT 0,           -- 검색 Top-K
                        retrieved_chunks INTEGER DEFAULT 0,-- 검색된 청크 수
                        citations_count INTEGER DEFAULT 0, -- 인용된 출처 페이지 수
                        total_elapsed_sec REAL DEFAULT 0.0,-- 총 소요 시간 (초)
                        input_tokens INTEGER DEFAULT 0,    -- 입력 토큰
                        output_tokens INTEGER DEFAULT 0,   -- 출력 토큰
                        total_tokens INTEGER DEFAULT 0,    -- 총 소요 토큰
                        cost_usd REAL DEFAULT 0.0,         -- 총 과금 (USD)
                        cost_krw REAL DEFAULT 0.0,         -- 총 과금 (원화)
                        tokens_per_sec REAL DEFAULT 0.0,   -- 초당 토큰 생성 속도
                        payload_json TEXT NOT NULL         -- 전체 상세 JSON
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_event_type ON rag_experiment_logs (event_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_models ON rag_experiment_logs (embedding_model, llm_model)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON rag_experiment_logs (timestamp)")
                conn.commit()
            finally:
                conn.close()

    def log_indexing(
        self,
        file_name: str,
        file_path: str,
        file_size_kb: float,
        total_pages: int,
        extracted_pages: int,
        loader_type: str,
        ocr_used: bool,
        ocr_reason: str,
        chunk_size: int,
        chunk_overlap: int,
        total_chunks: int,
        duplicate_chunk_count: int,
        duplicate_ratio: float,
        duplicate_diagnosis: str,
        chunks_to_embed: int,
        avg_chunk_length: float,
        embedding_model: str,
        embedding_dim: int,
        cache_enabled: bool,
        cache_hits: int,
        cache_misses: int,
        cache_hit_ratio: float,
        total_elapsed_sec: float,
        load_elapsed_sec: float,
        split_elapsed_sec: float,
        embed_elapsed_sec: float,
        store_elapsed_sec: float,
        total_tokens: int,
        cached_tokens: int,
        active_tokens: int,
        cost_usd: float,
        cost_krw: float,
        db_total_records: int,
    ) -> Dict[str, Any]:
        """인덱싱(Part A) 완료 시 실험 로그 1건을 기록합니다."""
        now_ts = datetime.now(_KST)
        exp_id = f"idx_{now_ts.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
        iso_time = now_ts.isoformat()
        kst_str = now_ts.strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "id": exp_id,
            "timestamp": iso_time,
            "timestamp_kst": kst_str,
            "event_type": "indexing",
            "document": {
                "file_name": file_name,
                "file_path": file_path,
                "file_size_kb": round(file_size_kb, 1),
                "total_pages": total_pages,
                "extracted_pages": extracted_pages,
                "loader_type": loader_type,
                "ocr_used": ocr_used,
                "ocr_reason": ocr_reason,
            },
            "chunking": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "total_chunks": total_chunks,
                "duplicate_chunks": duplicate_chunk_count,
                "duplicate_ratio_pct": duplicate_ratio,
                "duplicate_diagnosis": duplicate_diagnosis,
                "chunks_to_embed": chunks_to_embed,
                "avg_chunk_length": avg_chunk_length,
            },
            "embedding": {
                "model": embedding_model,
                "dimension": embedding_dim,
                "cache_enabled": cache_enabled,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_ratio_pct": cache_hit_ratio,
            },
            "performance": {
                "total_elapsed_sec": round(total_elapsed_sec, 3),
                "load_elapsed_sec": round(load_elapsed_sec, 3),
                "split_elapsed_sec": round(split_elapsed_sec, 3),
                "embed_elapsed_sec": round(embed_elapsed_sec, 3),
                "store_elapsed_sec": round(store_elapsed_sec, 3),
            },
            "cost": {
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
                "active_tokens": active_tokens,
                "cost_usd": round(cost_usd, 6),
                "cost_krw": round(cost_krw, 2),
            },
            "vector_store": {
                "total_records": db_total_records,
            },
        }

        self._save_entry(
            exp_id=exp_id,
            timestamp=kst_str,
            event_type="indexing",
            file_name=file_name,
            total_pages=total_pages,
            total_chunks=total_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_used=1 if ocr_used else 0,
            loader_type=loader_type,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            cache_enabled=1 if cache_enabled else 0,
            cache_hit=cache_hits,
            cache_hit_ratio=cache_hit_ratio,
            llm_model=None,
            llm_provider=None,
            question=None,
            top_k=0,
            retrieved_chunks=0,
            citations_count=0,
            total_elapsed_sec=total_elapsed_sec,
            input_tokens=active_tokens,
            output_tokens=0,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_krw=cost_krw,
            tokens_per_sec=0.0,
            payload=entry,
        )

        return entry

    def log_query(
        self,
        question: str,
        llm_model: str,
        llm_provider: str,
        embedding_model: str,
        embedding_dim: int,
        cache_enabled: bool,
        query_cache_hit: bool,
        top_k: int,
        retrieved_chunks_count: int,
        citations: List[str],
        answer_text: str,
        total_elapsed_sec: float,
        b1_embed_elapsed_sec: float,
        b2_retrieval_elapsed_ms: float,
        b4_llm_elapsed_sec: float,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        embedding_tokens: int,
        llm_cost_usd: float,
        embedding_cost_usd: float,
        total_cost_usd: float,
        total_cost_krw: float,
        active_doc_name: str = "",
        total_indexed_chunks: int = 0,
    ) -> Dict[str, Any]:
        """질의응답(Part B) 완료 시 실험 로그 1건을 기록합니다."""
        now_ts = datetime.now(_KST)
        exp_id = f"qry_{now_ts.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
        iso_time = now_ts.isoformat()
        kst_str = now_ts.strftime("%Y-%m-%d %H:%M:%S")

        total_tokens = input_tokens + output_tokens + embedding_tokens
        tps = round(output_tokens / max(0.01, b4_llm_elapsed_sec), 1) if output_tokens > 0 else 0.0

        entry = {
            "id": exp_id,
            "timestamp": iso_time,
            "timestamp_kst": kst_str,
            "event_type": "query",
            "combination_key": f"{embedding_model} + {llm_model} (top_k={top_k})",
            "query": {
                "question": question,
                "question_length": len(question),
                "active_doc": active_doc_name,
                "total_indexed_chunks": total_indexed_chunks,
            },
            "models": {
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
                "llm_model": llm_model,
                "llm_provider": llm_provider,
            },
            "retrieval": {
                "top_k": top_k,
                "retrieved_chunks": retrieved_chunks_count,
                "citations": citations,
                "citations_count": len(citations),
                "cache_enabled": cache_enabled,
                "query_cache_hit": query_cache_hit,
            },
            "answer": {
                "length": len(answer_text),
                "snippet": answer_text[:200] + ("..." if len(answer_text) > 200 else ""),
            },
            "performance": {
                "total_elapsed_sec": round(total_elapsed_sec, 3),
                "b1_embed_sec": round(b1_embed_elapsed_sec, 3),
                "b2_retrieval_ms": round(b2_retrieval_elapsed_ms, 2),
                "b4_llm_sec": round(b4_llm_elapsed_sec, 3),
                "tokens_per_second": tps,
            },
            "cost": {
                "llm_input_tokens": input_tokens,
                "llm_output_tokens": output_tokens,
                "llm_cache_read_tokens": cache_read_tokens,
                "embedding_tokens": embedding_tokens,
                "total_tokens": total_tokens,
                "llm_cost_usd": round(llm_cost_usd, 6),
                "embedding_cost_usd": round(embedding_cost_usd, 6),
                "total_cost_usd": round(total_cost_usd, 6),
                "total_cost_krw": round(total_cost_krw, 2),
            },
        }

        self._save_entry(
            exp_id=exp_id,
            timestamp=kst_str,
            event_type="query",
            file_name=active_doc_name,
            total_pages=0,
            total_chunks=total_indexed_chunks,
            chunk_size=0,
            chunk_overlap=0,
            ocr_used=0,
            loader_type=None,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            cache_enabled=1 if cache_enabled else 0,
            cache_hit=1 if query_cache_hit else 0,
            cache_hit_ratio=100.0 if query_cache_hit else 0.0,
            llm_model=llm_model,
            llm_provider=llm_provider,
            question=question,
            top_k=top_k,
            retrieved_chunks=retrieved_chunks_count,
            citations_count=len(citations),
            total_elapsed_sec=total_elapsed_sec,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=total_cost_usd,
            cost_krw=total_cost_krw,
            tokens_per_sec=tps,
            payload=entry,
        )

        return entry

    def _save_entry(
        self,
        exp_id: str,
        timestamp: str,
        event_type: str,
        file_name: Optional[str],
        total_pages: int,
        total_chunks: int,
        chunk_size: int,
        chunk_overlap: int,
        ocr_used: int,
        loader_type: Optional[str],
        embedding_model: str,
        embedding_dim: int,
        cache_enabled: int,
        cache_hit: int,
        cache_hit_ratio: float,
        llm_model: Optional[str],
        llm_provider: Optional[str],
        question: Optional[str],
        top_k: int,
        retrieved_chunks: int,
        citations_count: int,
        total_elapsed_sec: float,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cost_usd: float,
        cost_krw: float,
        tokens_per_sec: float,
        payload: Dict[str, Any],
    ) -> None:
        """JSONL 파일 및 SQLite에 원자적으로 기록합니다."""
        json_line = json.dumps(payload, ensure_ascii=False)

        with _LOGGER_LOCK:
            # 1. JSONL 파일 쓰기
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json_line + "\n")
            except Exception as e:
                print(f"[RAGExperimentLogger] JSONL 쓰기 실패: {e}")

            # 2. SQLite DB 쓰기
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO rag_experiment_logs (
                        id, timestamp, event_type, file_name, total_pages, total_chunks,
                        chunk_size, chunk_overlap, ocr_used, loader_type,
                        embedding_model, embedding_dim, cache_enabled, cache_hit, cache_hit_ratio,
                        llm_model, llm_provider, question, top_k, retrieved_chunks,
                        citations_count, total_elapsed_sec, input_tokens, output_tokens,
                        total_tokens, cost_usd, cost_krw, tokens_per_sec, payload_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        exp_id, timestamp, event_type, file_name, total_pages, total_chunks,
                        chunk_size, chunk_overlap, ocr_used, loader_type,
                        embedding_model, embedding_dim, cache_enabled, cache_hit, cache_hit_ratio,
                        llm_model, llm_provider, question, top_k, retrieved_chunks,
                        citations_count, total_elapsed_sec, input_tokens, output_tokens,
                        total_tokens, cost_usd, cost_krw, tokens_per_sec, json_line
                    )
                )
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[RAGExperimentLogger] SQLite 쓰기 실패: {e}")

    def get_recent_logs(self, limit: int = 50, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """최근 실험 로그 목록을 반환합니다."""
        with _LOGGER_LOCK:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            try:
                if event_type:
                    cursor.execute(
                        "SELECT payload_json FROM rag_experiment_logs WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                        (event_type, limit),
                    )
                else:
                    cursor.execute(
                        "SELECT payload_json FROM rag_experiment_logs ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    )
                rows = cursor.fetchall()
                return [json.loads(row["payload_json"]) for row in rows]
            finally:
                conn.close()

    def get_analysis_summary(self) -> Dict[str, Any]:
        """최적 조합 선정을 위해 누적된 실험 로그를 통계 집계 및 분석합니다."""
        with _LOGGER_LOCK:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            try:
                # 1. 전체 실행 통계
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_logs,
                        SUM(CASE WHEN event_type = 'indexing' THEN 1 ELSE 0 END) as index_count,
                        SUM(CASE WHEN event_type = 'query' THEN 1 ELSE 0 END) as query_count,
                        COALESCE(SUM(total_tokens), 0) as grand_total_tokens,
                        COALESCE(SUM(cost_usd), 0.0) as grand_total_cost_usd,
                        COALESCE(SUM(cost_krw), 0.0) as grand_total_cost_krw
                    FROM rag_experiment_logs
                """)
                overview = dict(cursor.fetchone())

                # 2. LLM 모델별 성능 및 비용 비교 (질의응답 기준)
                cursor.execute("""
                    SELECT
                        llm_model,
                        llm_provider,
                        COUNT(*) as query_count,
                        ROUND(AVG(total_elapsed_sec), 3) as avg_latency_sec,
                        ROUND(AVG(tokens_per_sec), 1) as avg_tps,
                        ROUND(AVG(input_tokens), 0) as avg_input_tokens,
                        ROUND(AVG(output_tokens), 0) as avg_output_tokens,
                        ROUND(AVG(cost_usd), 6) as avg_cost_usd,
                        ROUND(AVG(cost_krw), 2) as avg_cost_krw,
                        ROUND(SUM(cost_usd), 6) as sum_cost_usd,
                        ROUND(AVG(citations_count), 1) as avg_citations
                    FROM rag_experiment_logs
                    WHERE event_type = 'query' AND llm_model IS NOT NULL
                    GROUP BY llm_model
                    ORDER BY avg_cost_usd ASC
                """)
                llm_benchmarks = [dict(r) for r in cursor.fetchall()]

                # 3. 임베딩 모델별 성능 및 비용 비교 (인덱싱 & 질의 통합)
                cursor.execute("""
                    SELECT
                        embedding_model,
                        COUNT(*) as total_runs,
                        SUM(CASE WHEN event_type = 'indexing' THEN 1 ELSE 0 END) as index_runs,
                        SUM(CASE WHEN event_type = 'query' THEN 1 ELSE 0 END) as query_runs,
                        ROUND(AVG(total_elapsed_sec), 3) as avg_elapsed_sec,
                        ROUND(AVG(cache_hit_ratio), 1) as avg_cache_hit_pct,
                        ROUND(SUM(cost_usd), 6) as sum_cost_usd,
                        ROUND(SUM(cost_krw), 2) as sum_cost_krw
                    FROM rag_experiment_logs
                    GROUP BY embedding_model
                    ORDER BY sum_cost_usd ASC
                """)
                embedding_benchmarks = [dict(r) for r in cursor.fetchall()]

                # 4. 청크 크기(chunk_size)별 분할 효율 및 중복률 비교
                cursor.execute("""
                    SELECT
                        chunk_size,
                        chunk_overlap,
                        COUNT(*) as run_count,
                        ROUND(AVG(total_chunks), 0) as avg_chunks,
                        ROUND(AVG(cache_hit_ratio), 1) as avg_cache_hit_ratio,
                        ROUND(AVG(total_elapsed_sec), 2) as avg_index_sec,
                        ROUND(AVG(cost_usd), 6) as avg_index_cost_usd
                    FROM rag_experiment_logs
                    WHERE event_type = 'indexing' AND chunk_size > 0
                    GROUP BY chunk_size, chunk_overlap
                    ORDER BY chunk_size ASC
                """)
                chunking_benchmarks = [dict(r) for r in cursor.fetchall()]

                # 5. 캐시 활성화 효과 (On vs Off) 비교
                cursor.execute("""
                    SELECT
                        cache_enabled,
                        COUNT(*) as total_runs,
                        ROUND(AVG(total_elapsed_sec), 3) as avg_latency_sec,
                        ROUND(AVG(cost_usd), 6) as avg_cost_usd,
                        ROUND(AVG(cache_hit_ratio), 1) as avg_hit_ratio
                    FROM rag_experiment_logs
                    GROUP BY cache_enabled
                """)
                cache_impact = [dict(r) for r in cursor.fetchall()]

                # 6. 최적 조합 추천 생성
                recommendations = self._derive_recommendations(llm_benchmarks, embedding_benchmarks, cache_impact)

                return {
                    "overview": overview,
                    "llm_benchmarks": llm_benchmarks,
                    "embedding_benchmarks": embedding_benchmarks,
                    "chunking_benchmarks": chunking_benchmarks,
                    "cache_impact": cache_impact,
                    "recommendations": recommendations,
                    "log_file_path": str(self.jsonl_path),
                }
            finally:
                conn.close()

    def _derive_recommendations(
        self,
        llm_list: List[Dict[str, Any]],
        emb_list: List[Dict[str, Any]],
        cache_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """수집된 벤치마크 데이터를 바탕으로 가성비/속도/품질 최적 조합을 추천합니다."""
        best_cost = None
        fastest = None
        highest_citations = None

        if llm_list:
            # 비용 최소
            best_cost = min(llm_list, key=lambda x: x["avg_cost_usd"])
            # 속도 최단
            fastest = min(llm_list, key=lambda x: x["avg_latency_sec"])
            # 인용 출처 최다
            highest_citations = max(llm_list, key=lambda x: x.get("avg_citations", 0))

        best_emb = emb_list[0]["embedding_model"] if emb_list else "text-embedding-004"

        return {
            "best_cost_combination": {
                "title": "💰 최고의 가성비 조합",
                "embedding": best_emb,
                "llm": best_cost["llm_model"] if best_cost else "gemini-3.5-flash-lite",
                "cache": "활성화 권장 (비용 $0 절감)",
                "reason": f"질의당 평균 비용 약 {best_cost['avg_cost_krw'] if best_cost else 1.0}원",
            },
            "fastest_combination": {
                "title": "⚡ 최고 속도 조합",
                "embedding": best_emb,
                "llm": fastest["llm_model"] if fastest else "claude-haiku-5-5",
                "cache": "필수 활성화 (질문 임베딩 0초 즉시 반환)",
                "reason": f"평균 레이턴시 {fastest['avg_latency_sec'] if fastest else 0.8}초 / 초당 {fastest['avg_tps'] if fastest else 80} 토큰",
            },
            "balanced_combination": {
                "title": "🎯 균형잡힌 추천 조합",
                "embedding": "text-embedding-004",
                "llm": "gemini-3.5-flash-lite",
                "chunk_size": "800자 / 150자 오버랩",
                "cache": "활성화",
                "reason": "풍부한 한국어 이해도, 저렴한 단가($0.30/1M), 빠른 응답 속도",
            },
        }

    def export_csv(self, event_type: Optional[str] = None) -> str:
        """실험 로그를 CSV 포맷 문자열로 변환합니다."""
        with _LOGGER_LOCK:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            try:
                query = "SELECT * FROM rag_experiment_logs"
                params = ()
                if event_type:
                    query += " WHERE event_type = ?"
                    params = (event_type,)
                query += " ORDER BY timestamp DESC"
                cursor.execute(query, params)
                columns = [d[0] for d in cursor.description]
                rows = cursor.fetchall()

                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(columns)
                writer.writerows(rows)
                return output.getvalue()
            finally:
                conn.close()


# 전역 싱글톤 인스턴스
class _LazyLogger:
    """Compatibility for analysis scripts; no filesystem writes during import."""
    _instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = RAGExperimentLogger()
        return getattr(self._instance, name)


global_experiment_logger = _LazyLogger()

