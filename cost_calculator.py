"""Vertex AI Model Garden & Partner Models 요율 기반 토큰 카운터 및 과금 계산 모듈.

참조 공식 가이드:
1. Google First-Party & Open-Weights MaaS 요율:
   https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform
2. Partner Models (Anthropic Claude on Vertex AI) 요율:
   https://cloud.google.com/model-garden/partner-models/pricing
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# 기본 원/달러 환율 (KRW per USD)
DEFAULT_EXCHANGE_RATE = 1400.0

# =============================================================================
# 실습용 예시 가격 카탈로그 (단위: 1,000,000 토큰 = 1M Tokens 기준 USD)
# =============================================================================
PRICING_CATALOG: Dict[str, Dict[str, Any]] = {
    "gemini-3.8-flash": {
        "name": "Gemini 3.8 Flash (Vertex AI)",
        "provider": "Google (Vertex AI)",
        "category": "Google First-Party",
        "input_price_per_1m": 0.75,       # $0.75 / 1M tokens ($0.00075 / 1k)
        "output_price_per_1m": 3.75,      # $3.75 / 1M tokens ($0.00375 / 1k)
        "cache_read_price_per_1m": 0.1875,# $0.1875 / 1M tokens
        "unit": "1M tokens",
        "notes": "프로모션 가격 (2026.12.31까지), 1M Context Window, 복합 추론 최적화",
        "official_source": "https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform",
    },
    "gemini-3.5-flash-lite": {
        "name": "Gemini 3.5 Flash-Lite (Vertex AI)",
        "provider": "Google (Vertex AI)",
        "category": "Google First-Party",
        "input_price_per_1m": 0.30,       # $0.30 / 1M tokens ($0.00030 / 1k)
        "output_price_per_1m": 2.50,      # $2.50 / 1M tokens ($0.00250 / 1k)
        "cache_read_price_per_1m": 0.075, # $0.075 / 1M tokens
        "unit": "1M tokens",
        "notes": "초경량·초저지연 플래시 라이트, 1M Context Window, 고효율 RAG 최적화",
        "official_source": "https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform",
    },
    "deepseek-ai/deepseek-v4.1-flash-maas": {
        "name": "DeepSeek V4.1 Flash (Vertex AI MaaS)",
        "provider": "DeepSeek (Model Garden MaaS)",
        "category": "Partner / Open-Weights MaaS",
        "input_price_per_1m": 0.30,       # $0.30 / 1M tokens ($0.00030 / 1k)
        "output_price_per_1m": 1.20,      # $1.20 / 1M tokens ($0.00120 / 1k)
        "cache_read_price_per_1m": 0.03,  # $0.03 / 1M tokens
        "unit": "1M tokens",
        "notes": "Vertex AI Model Garden 관리형 MaaS 엔드포인트, 온디맨드 서버리스 서빙",
        "official_source": "https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform",
    },
    "google/gemma-4-26b-a4b-it-maas": {
        "name": "Gemma 4 26B (Vertex AI MaaS)",
        "provider": "Google (Model Garden MaaS)",
        "category": "Google Open-Weights MaaS",
        "input_price_per_1m": 0.15,       # $0.15 / 1M tokens ($0.00015 / 1k)
        "output_price_per_1m": 0.60,      # $0.60 / 1M tokens ($0.00060 / 1k)
        "cache_read_price_per_1m": 0.015, # $0.015 / 1M tokens
        "unit": "1M tokens",
        "notes": "Gemma 4 26B-A4B-IT Thinking 지원 MaaS 엔드포인트, 고효율 오픈모델",
        "official_source": "https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform",
    },
    "claude-haiku-5-5": {
        "name": "Claude Haiku 5.5 (Vertex AI)",
        "provider": "Anthropic on Vertex AI",
        "category": "Partner Models",
        "input_price_per_1m": 0.10,       # $0.10 / 1M tokens (<= 200k)
        "output_price_per_1m": 0.50,      # $0.50 / 1M tokens (<= 100k)
        "cache_read_price_per_1m": 0.01,  # $0.01 / 1M tokens (Prompt Caching)
        "unit": "1M tokens",
        "notes": "Anthropic on Vertex AI 파트너 모델, 최고 속도 및 고효율 분석",
        "official_source": "https://cloud.google.com/model-garden/partner-models/pricing",
    },
    "gemma4:e2b": {
        "name": "Gemma 4 E2B (Ollama 로컬)",
        "provider": "Ollama (로컬 하드웨어)",
        "category": "Local Self-Hosted",
        "input_price_per_1m": 0.0,
        "output_price_per_1m": 0.0,
        "cache_read_price_per_1m": 0.0,
        "unit": "무료 (온디바이스)",
        "notes": "사용자 개인 컴퓨터 GPU/CPU 로컬 실행 (클라우드 API 과금 $0)",
        "official_source": "로컬 온디바이스 실행",
    },
    "text-embedding-004": {
        "name": "Text Embedding 004 (Vertex AI)",
        "provider": "Google (Vertex AI)",
        "category": "Embeddings",
        "input_price_per_1m": 0.10,       # $0.10 / 1M tokens ($0.025 / 1k chars)
        "output_price_per_1m": 0.0,
        "cache_read_price_per_1m": 0.0,
        "unit": "1M tokens",
        "notes": "RAG 벡터 임베딩 (768차원)",
        "official_source": "https://cloud.google.com/model-garden/pricing",
    },
    "gemini-embedding-2": {
        "name": "Gemini Embedding 2 (Vertex AI)",
        "provider": "Google (Vertex AI)",
        "category": "Embeddings",
        "input_price_per_1m": 0.20,       # $0.20 / 1M tokens
        "output_price_per_1m": 0.0,
        "cache_read_price_per_1m": 0.0,
        "unit": "1M tokens",
        "notes": "고차원 멀티모달 임베딩 (3,072차원)",
        "official_source": "https://cloud.google.com/model-garden/pricing",
    },
}


# These are educational estimates, not verified current billing rates.
for _price in PRICING_CATALOG.values():
    _price["notes"] = "실습용 예시 요율. 최신 가격·계약·청구액과 다를 수 있습니다."


def normalize_model_id(model_name: str) -> str:
    """별칭이나 부분 모델명을 정규 카탈로그 ID로 매핑합니다."""
    lower = model_name.strip().lower()
    if lower in PRICING_CATALOG:
        return lower
    if any(k in lower for k in ("claude", "haiku")):
        return "claude-haiku-5-5"
    if "deepseek" in lower:
        return "deepseek-ai/deepseek-v4.1-flash-maas"
    if ":" in lower or "ollama" in lower:
        return "gemma4:e2b"
    if "gemma" in lower:
        return "google/gemma-4-26b-a4b-it-maas"
    if "lite" in lower:
        return "gemini-3.5-flash-lite"
    if "flash" in lower or "gemini" in lower:
        return "gemini-3.8-flash"
    if "embedding-2" in lower:
        return "gemini-embedding-2"
    if "embedding" in lower:
        return "text-embedding-004"
    return "gemini-3.8-flash"


def extract_tokens_from_response(
    response: Any,
    prompt_text: str = "",
    answer_text: str = "",
) -> Dict[str, int]:
    """LangChain 응답 객체에서 실제 토큰 수를 추출하고 필요 시 폴백 계산합니다."""
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0

    # 1. usage_metadata 확인 (ChatGoogleGenerativeAI, ChatAnthropicVertex, ChatOpenAI 공통)
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens") or 0
        output_tokens = usage.get("output_tokens") or 0
        cache_details = usage.get("input_token_details", {})
        if isinstance(cache_details, dict):
            cache_read_tokens = cache_details.get("cache_read") or 0

    # 2. response_metadata 폴백 확인
    if input_tokens == 0 and output_tokens == 0:
        resp_meta = getattr(response, "response_metadata", {})
        if isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
            if isinstance(token_usage, dict):
                input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
                output_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
                cache_read_tokens = token_usage.get("cache_read_input_tokens") or 0

    # 3. 휴리스틱 폴백 추정 (문자열 길이 기반)
    # 한글/영어 혼합 텍스트: 약 2.2자 ≈ 1토큰
    if input_tokens <= 0 and prompt_text:
        input_tokens = max(1, int(len(prompt_text) / 2.2))
    if output_tokens <= 0 and answer_text:
        output_tokens = max(1, int(len(answer_text) / 2.2))

    total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": total_tokens,
    }


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    exchange_rate: float = DEFAULT_EXCHANGE_RATE,
) -> Dict[str, Any]:
    """실습용 예시 요율에 따라 특정 모델의 요청 비용(USD 및 KRW)을 계산합니다."""
    norm_id = normalize_model_id(model_name)
    pricing = PRICING_CATALOG.get(norm_id, PRICING_CATALOG["gemini-3.8-flash"])

    in_rate = pricing["input_price_per_1m"]
    out_rate = pricing["output_price_per_1m"]
    cache_rate = pricing["cache_read_price_per_1m"]

    # 캐시된 토큰은 일반 입력 토큰에서 제외하여 감면된 요율 적용
    normal_input = max(0, input_tokens - cache_read_tokens)

    cost_input_usd = (normal_input / 1_000_000.0) * in_rate
    cost_cache_usd = (cache_read_tokens / 1_000_000.0) * cache_rate
    cost_output_usd = (output_tokens / 1_000_000.0) * out_rate
    total_cost_usd = cost_input_usd + cost_cache_usd + cost_output_usd

    total_cost_krw = total_cost_usd * exchange_rate

    return {
        "model_id": norm_id,
        "model_name": pricing["name"],
        "provider": pricing["provider"],
        "category": pricing["category"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": input_tokens + output_tokens,
        "rates": {
            "input_per_1m": in_rate,
            "output_per_1m": out_rate,
            "cache_read_per_1m": cache_rate,
        },
        "cost_input_usd": round(cost_input_usd, 6),
        "cost_cache_usd": round(cost_cache_usd, 6),
        "cost_output_usd": round(cost_output_usd, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "total_cost_krw": round(total_cost_krw, 3),
        "formatted_usd": f"${total_cost_usd:.6f}",
        "formatted_krw": f"약 {total_cost_krw:.2f}원",
        "official_source": pricing["official_source"],
    }


def calculate_embedding_cost(
    model_name: str,
    char_count_or_tokens: int,
    cached: bool = False,
    exchange_rate: float = DEFAULT_EXCHANGE_RATE,
) -> Dict[str, Any]:
    """RAG 질문/문서 임베딩 비용을 계산합니다 (캐시 적중 시 $0)."""
    norm_id = "gemini-embedding-2" if "2" in model_name else "text-embedding-004"
    pricing = PRICING_CATALOG.get(norm_id, PRICING_CATALOG["text-embedding-004"])

    if cached:
        return {
            "model_id": norm_id,
            "model_name": pricing["name"],
            "cached": True,
            "cost_usd": 0.0,
            "cost_krw": 0.0,
            "formatted_usd": "$0.000000 (캐시 적중)",
            "formatted_krw": "0.00원 (캐시 적중)",
        }

    # 토큰 추정 (약 2.5자 = 1토큰)
    est_tokens = max(1, int(char_count_or_tokens / 2.5))
    cost_usd = (est_tokens / 1_000_000.0) * pricing["input_price_per_1m"]
    cost_krw = cost_usd * exchange_rate

    return {
        "model_id": norm_id,
        "model_name": pricing["name"],
        "cached": False,
        "estimated_tokens": est_tokens,
        "cost_usd": round(cost_usd, 6),
        "cost_krw": round(cost_krw, 3),
        "formatted_usd": f"${cost_usd:.6f}",
        "formatted_krw": f"약 {cost_krw:.2f}원",
    }


class CostTracker:
    """세션 및 실시간 RAG 질의 누적 토큰과 과금 현황을 추적하는 스레드 안전 싱글톤."""

    def __init__(self, exchange_rate: float = DEFAULT_EXCHANGE_RATE):
        self._lock = threading.Lock()
        self.exchange_rate = exchange_rate
        self.start_time = time.time()
        self.total_queries = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read_tokens = 0
        self.total_cost_usd = 0.0
        self.total_cost_krw = 0.0

        # 모델별 누적 현황
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        for m_id, info in PRICING_CATALOG.items():
            self.model_stats[m_id] = {
                "name": info["name"],
                "provider": info["provider"],
                "category": info["category"],
                "queries": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_krw": 0.0,
            }

        # 최근 질의 이력 (최대 30개)
        self.recent_history: List[Dict[str, Any]] = []

    def record_query(
        self,
        model_name: str,
        question: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        elapsed_sec: float = 0.0,
        embedding_cost_usd: float = 0.0,
    ) -> Dict[str, Any]:
        """질의 1건의 토큰 및 과금을 기록하고 세션 통계를 갱신합니다."""
        norm_id = normalize_model_id(model_name)
        cost_info = calculate_cost(
            norm_id, input_tokens, output_tokens, cache_read_tokens, self.exchange_rate
        )

        llm_cost_usd = cost_info["total_cost_usd"]
        combined_total_usd = llm_cost_usd + embedding_cost_usd
        combined_total_krw = combined_total_usd * self.exchange_rate

        with self._lock:
            self.total_queries += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cache_read_tokens += cache_read_tokens
            self.total_cost_usd += llm_cost_usd
            self.total_cost_krw += (llm_cost_usd * self.exchange_rate)

            if norm_id not in self.model_stats:
                self.model_stats[norm_id] = {
                    "name": cost_info["model_name"],
                    "provider": cost_info["provider"],
                    "category": cost_info["category"],
                    "queries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_krw": 0.0,
                }

            m_stat = self.model_stats[norm_id]
            m_stat["queries"] += 1
            m_stat["input_tokens"] += input_tokens
            m_stat["output_tokens"] += output_tokens
            m_stat["cache_read_tokens"] += cache_read_tokens
            m_stat["total_tokens"] += (input_tokens + output_tokens)
            m_stat["cost_usd"] = round(m_stat["cost_usd"] + llm_cost_usd, 6)
            m_stat["cost_krw"] = round(m_stat["cost_krw"] + cost_info["total_cost_krw"], 3)

            history_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "model_id": norm_id,
                "model_name": cost_info["model_name"],
                "question": question[:60] + ("..." if len(question) > 60 else ""),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": round(combined_total_usd, 6),
                "cost_krw": round(combined_total_krw, 2),
                "elapsed_sec": elapsed_sec,
            }
            self.recent_history.insert(0, history_entry)
            if len(self.recent_history) > 30:
                self.recent_history.pop()

        return {
            "query_cost": cost_info,
            "embedding_cost_usd": round(embedding_cost_usd, 6),
            "combined_total_usd": round(combined_total_usd, 6),
            "combined_total_krw": round(combined_total_krw, 3),
        }

    def record_embedding_job(
        self,
        model_name: str,
        total_tokens_or_chars: int,
        cached_tokens_or_chars: int = 0,
        job_type: str = "index",
        doc_name: str = "",
    ) -> Dict[str, Any]:
        """문서 인덱싱(Part A) 또는 질의(Part B) 임베딩 토큰 및 과금을 기록합니다."""
        norm_id = "gemini-embedding-2" if "2" in model_name else "text-embedding-004"
        pricing = PRICING_CATALOG.get(norm_id, PRICING_CATALOG["text-embedding-004"])

        # 비캐시 토큰 계산 (약 2.5자 = 1토큰 기준)
        active_chars = max(0, total_tokens_or_chars - cached_tokens_or_chars)
        est_tokens = max(1, int(active_chars / 2.5)) if active_chars > 0 else 0
        cached_est_tokens = int(cached_tokens_or_chars / 2.5) if cached_tokens_or_chars > 0 else 0
        total_est_tokens = est_tokens + cached_est_tokens

        # 임베딩 단가 ($0.10 / 1M tokens)
        cost_usd = (est_tokens / 1_000_000.0) * pricing["input_price_per_1m"]
        cost_krw = cost_usd * self.exchange_rate

        with self._lock:
            self.total_input_tokens += total_est_tokens
            self.total_cache_read_tokens += cached_est_tokens
            self.total_cost_usd += cost_usd
            self.total_cost_krw += cost_krw

            if norm_id not in self.model_stats:
                self.model_stats[norm_id] = {
                    "name": pricing["name"],
                    "provider": pricing["provider"],
                    "category": pricing["category"],
                    "queries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "cost_krw": 0.0,
                }

            m_stat = self.model_stats[norm_id]
            m_stat["queries"] += 1
            m_stat["input_tokens"] += total_est_tokens
            m_stat["cache_read_tokens"] += cached_est_tokens
            m_stat["total_tokens"] += total_est_tokens
            m_stat["cost_usd"] = round(m_stat["cost_usd"] + cost_usd, 6)
            m_stat["cost_krw"] = round(m_stat["cost_krw"] + cost_krw, 3)

            label = f"[문서 인덱싱] {doc_name}" if job_type == "index" else f"[질문 임베딩] {doc_name}"
            history_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "model_id": norm_id,
                "model_name": pricing["name"],
                "question": label[:60],
                "input_tokens": total_est_tokens,
                "output_tokens": 0,
                "total_tokens": total_est_tokens,
                "cost_usd": round(cost_usd, 6),
                "cost_krw": round(cost_krw, 2),
                "elapsed_sec": 0.0,
            }
            self.recent_history.insert(0, history_entry)
            if len(self.recent_history) > 30:
                self.recent_history.pop()

        return {
            "model_id": norm_id,
            "model_name": pricing["name"],
            "total_tokens": total_est_tokens,
            "cached_tokens": cached_est_tokens,
            "active_tokens": est_tokens,
            "cost_usd": round(cost_usd, 6),
            "cost_krw": round(cost_krw, 3),
            "formatted_usd": f"${cost_usd:.6f}",
            "formatted_krw": f"약 {cost_krw:.2f}원",
        }

    def record_ocr_job(self, model_name: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        result = self.record_query(model_name, "[OCR 페이지 처리]", input_tokens, output_tokens)
        with self._lock:
            self.total_queries -= 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """현재까지의 누적 세션 과금 현황을 반환합니다 (LLM 및 임베딩 모델 모두 포함)."""
        with self._lock:
            # LLM 및 임베딩 모델 모두 포함하여 통계 반환
            active_models = [
                {"id": k, **v}
                for k, v in self.model_stats.items()
            ]
            # 사용된 모델을 상단에, 비용 및 사용 횟수 내림차순 정렬
            active_models.sort(
                key=lambda x: (1 if x["queries"] > 0 else 0, x["cost_usd"], x["queries"]),
                reverse=True,
            )

            return {
                "estimate_only": True,
                "scope_notice": "실습용 예시 요율·추정 토큰입니다. OCR 사용량을 포함하며 SDK 재시도·실제 청구액은 Cloud Billing과 대조하세요.",
                "exchange_rate": self.exchange_rate,
                "uptime_minutes": round((time.time() - self.start_time) / 60, 1),
                "total_queries": self.total_queries,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cache_read_tokens": self.total_cache_read_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "total_cost_krw": round(self.total_cost_krw, 2),
                "formatted_total_usd": f"${self.total_cost_usd:.4f}",
                "formatted_total_krw": f"약 {self.total_cost_krw:,.1f}원",
                "models": active_models,
                "recent_history": list(self.recent_history),
            }

    def reset(self) -> None:
        """세션 누적 통계를 초기화합니다."""
        with self._lock:
            self.start_time = time.time()
            self.total_queries = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cache_read_tokens = 0
            self.total_cost_usd = 0.0
            self.total_cost_krw = 0.0
            for m in self.model_stats.values():
                m["queries"] = 0
                m["input_tokens"] = 0
                m["output_tokens"] = 0
                m["cache_read_tokens"] = 0
                m["total_tokens"] = 0
                m["cost_usd"] = 0.0
                m["cost_krw"] = 0.0
            self.recent_history.clear()


# 전역 싱글톤 인스턴스
global_cost_tracker = CostTracker()


def get_pricing_matrix() -> List[Dict[str, Any]]:
    """UI 및 API에서 참조할 공식 요율표 전체 목록을 반환합니다."""
    results = []
    for m_id, info in PRICING_CATALOG.items():
        results.append({
            "id": m_id,
            "name": info["name"],
            "provider": info["provider"],
            "category": info["category"],
            "input_price_per_1m": info["input_price_per_1m"],
            "output_price_per_1m": info["output_price_per_1m"],
            "cache_read_price_per_1m": info["cache_read_price_per_1m"],
            "input_per_1k": round(info["input_price_per_1m"] / 1000.0, 6),
            "output_per_1k": round(info["output_price_per_1m"] / 1000.0, 6),
            "unit": info["unit"],
            "notes": info["notes"],
            "official_source": info["official_source"],
        })
    return results


def simulate_all_models(
    input_tokens: int,
    output_tokens: int,
    query_count: int = 1,
    exchange_rate: float = DEFAULT_EXCHANGE_RATE,
) -> List[Dict[str, Any]]:
    """동일한 토큰 사용량에 대해 모든 지원 모델(LLM 및 임베딩)의 예상 비용을 비교 시뮬레이션합니다."""
    models_to_simulate = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "deepseek-ai/deepseek-v4.1-flash-maas",
        "google/gemma-4-26b-a4b-it-maas",
        "claude-haiku-5-5",
        "gemma4:e2b",
        "text-embedding-004",
        "gemini-embedding-2",
    ]

    comparisons = []
    for m_id in models_to_simulate:
        single = calculate_cost(m_id, input_tokens, output_tokens, exchange_rate=exchange_rate)
        total_usd = single["total_cost_usd"] * query_count
        total_krw = single["total_cost_krw"] * query_count
        comparisons.append({
            "id": m_id,
            "name": single["model_name"],
            "provider": single["provider"],
            "category": single["category"],
            "single_query_usd": single["total_cost_usd"],
            "single_query_krw": single["total_cost_krw"],
            "total_usd": round(total_usd, 6),
            "total_krw": round(total_krw, 2),
            "formatted_usd": f"${total_usd:.6f}",
            "formatted_krw": f"약 {total_krw:,.1f}원",
            "rates": single["rates"],
        })

    # 비용 오름차순 정렬 (가장 경제적인 순)
    comparisons.sort(key=lambda x: x["total_usd"])
    return comparisons

