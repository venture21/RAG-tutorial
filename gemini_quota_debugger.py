"""Gemini 및 Google Cloud API 호출 시 발생하는 429 (Resource Exhausted) 에러 분석 모듈.

429 에러 응답 페이로드(JSON/Protobuf/Exception)를 분석하여:
1. 프로젝트 리소스(쿼터) 고갈: QuotaFailure 객체 또는 quota_metric이 명시된 경우
2. 구글 클라우드 리전 리소스 고갈 / DSQ 순간 급증 한도(트래픽 버스트):
   구체적인 메트릭 없이 재시도를 권고하거나 일시적인 용량 부족인 경우
를 명확히 구분하고 상세한 디버깅 로그 및 안내 메시지를 제공합니다.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gemini_quota_debugger")


def parse_error_payload(exc: Exception) -> Tuple[Optional[int], Dict[str, Any], str]:
    """예외 객체에서 HTTP 상태 코드, 페이로드 딕셔너리(JSON), 전체 메시지를 추출합니다."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    raw_message = str(exc)
    payload: Dict[str, Any] = {}

    # 1. google.genai.errors.APIError 의 details 속성 확인
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        payload = details
    elif isinstance(details, list):
        payload = {"details": details}

    # 2. exception의 response 속성 (requests, httpx, google-api-core)
    resp = getattr(exc, "response", None) or getattr(exc, "_response", None)
    if resp is not None:
        try:
            if hasattr(resp, "json") and callable(resp.json):
                payload = resp.json()
            elif hasattr(resp, "text"):
                payload = json.loads(resp.text)
        except Exception:
            pass

    # 3. 문자열 내 JSON 파싱 시도 (메시지 본문에 JSON 문자열이 포함된 경우)
    if not payload and "{" in raw_message and "}" in raw_message:
        start_idx = raw_message.find("{")
        end_idx = raw_message.rfind("}") + 1
        candidate = raw_message[start_idx:end_idx]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            pass

    # HTTP 상태 코드 보정
    if code is None:
        if "429" in raw_message or "ResourceExhausted" in type(exc).__name__:
            code = 429

    return code, payload, raw_message


def analyze_gemini_429_error(exc: Exception) -> Dict[str, Any]:
    """Gemini 429 에러의 근본 원인을 분석하여 진단 결과를 반환합니다.

    Returns:
        {
            "is_429": bool,
            "cause_type": "PROJECT_QUOTA_EXHAUSTED" | "CLOUD_REGION_RESOURCE_EXHAUSTED_OR_BURST" | "UNKNOWN_429",
            "title": str,
            "summary": str,
            "quota_metrics": List[str],
            "retry_delay_seconds": Optional[float],
            "details_entries": List[Any],
            "debug_message": str,
            "recommended_action": str,
        }
    """
    code, payload, raw_message = parse_error_payload(exc)

    # 429 / ResourceExhausted 여부 판별
    is_429 = (
        code == 429
        or "429" in raw_message
        or "RESOURCE_EXHAUSTED" in raw_message.upper()
        or "ResourceExhausted" in type(exc).__name__
    )

    if not is_429:
        return {
            "is_429": False,
            "cause_type": "NOT_429",
            "title": f"HTTP {code or 'Error'}",
            "summary": raw_message[:200],
            "quota_metrics": [],
            "retry_delay_seconds": None,
            "details_entries": [],
            "debug_message": f"429 에러가 아닙니다: {raw_message}",
            "recommended_action": "에러 메시지를 확인하세요.",
        }

    # error.details 배열 탐색
    error_obj = payload.get("error", payload) if isinstance(payload, dict) else {}
    details_list: List[Any] = []
    if isinstance(error_obj, dict) and "details" in error_obj:
        details_candidate = error_obj["details"]
        if isinstance(details_candidate, list):
            details_list = details_candidate
        elif isinstance(details_candidate, dict):
            details_list = [details_candidate]

    # 구체적인 QuotaFailure 및 quota_metric 검출
    quota_metrics: List[str] = []
    has_quota_failure = False
    retry_delay_seconds: Optional[float] = None

    payload_str = json.dumps(payload, ensure_ascii=False) if payload else raw_message

    for detail in details_list:
        if not isinstance(detail, dict):
            continue
        type_str = detail.get("@type", "")

        # QuotaFailure 감지
        if "QuotaFailure" in type_str or "violations" in detail:
            has_quota_failure = True
            for violation in detail.get("violations", []):
                metric = violation.get("quotaMetric") or violation.get("quota_metric")
                desc = violation.get("description", "")
                subject = violation.get("subject", "")
                if metric:
                    quota_metrics.append(metric)
                elif "quota_metric:" in desc:
                    parts = desc.split("quota_metric:")
                    if len(parts) > 1:
                        quota_metrics.append(parts[1].split()[0].strip())
                elif desc:
                    quota_metrics.append(f"{subject}: {desc}")

        # RetryInfo (지연 시간 권고)
        if "RetryInfo" in type_str or "retryDelay" in detail:
            delay_str = str(detail.get("retryDelay", "")).replace("s", "")
            try:
                retry_delay_seconds = float(delay_str)
            except Exception:
                pass

        # ErrorInfo 감지
        if "ErrorInfo" in type_str:
            metadata = detail.get("metadata", {})
            if "quota_metric" in metadata:
                quota_metrics.append(metadata["quota_metric"])

    # 텍스트 내 quota_metric 키워드 검출 (페이로드가 중첩되지 않고 텍스트로만 온 경우)
    if "quota_metric" in payload_str.lower() or "quotafailure" in payload_str.lower():
        has_quota_failure = True

    # -------------------------------------------------------------------------
    # 원인 판정 로직
    # -------------------------------------------------------------------------
    if has_quota_failure or quota_metrics:
        cause_type = "PROJECT_QUOTA_EXHAUSTED"
        title = "🔴 [프로젝트 리소스 한도 초과] Project Quota Exhausted"
        metrics_str = ", ".join(quota_metrics) if quota_metrics else "세부 쿼터 메트릭 지정됨"
        summary = (
            f"해당 Google Cloud 프로젝트의 Gemini API 할당량(Quota)이 소진되었습니다. "
            f"(초과 메트릭: {metrics_str})"
        )
        debug_msg = (
            f"\n"
            f"================================================================================\n"
            f"🔍 [429 분석 결과: 프로젝트 리소스 한도 초과 (Project Quota)]\n"
            f"================================================================================\n"
            f"• 판정 이유: error.details 내에 QuotaFailure 객체 또는 quota_metric이 명시됨\n"
            f"• 초과된 쿼터 메트릭: {metrics_str}\n"
            f"• 에러 코드: 429 RESOURCE_EXHAUSTED\n"
            f"• 권장 조치사항:\n"
            f"  1) Google Cloud Console -> 'IAM 및 관리자' -> '할당량(Quotas)' 메뉴 이동\n"
            f"  2) 해당 메트릭({metrics_str})의 분당 요청 수(RPM) 또는 분당 토큰 수(TPM) 한도 상향 요청\n"
            f"  3) 동시 요청 수를 줄이거나 호출 간 지연 시간(Rate Limiter)을 설정\n"
            f"================================================================================"
        )
        action = "Google Cloud Console의 '할당량' 메뉴에서 해당 API 메트릭의 한도 상향을 요청하거나 요청 빈도를 낮추세요."
    else:
        cause_type = "CLOUD_REGION_RESOURCE_EXHAUSTED_OR_BURST"
        title = "🟡 [구글 클라우드 리전 리소스 고갈 / DSQ 순간 트래픽 제한]"
        delay_info = f"권장 대기시간: {retry_delay_seconds}초" if retry_delay_seconds else "지수 백오프(Exponential Backoff) 재시도 필요"
        summary = (
            f"구체적인 quota_metric 없이 재시도를 권고하는 응답으로, "
            f"해당 리전의 순간적인 서버 용량 부족(클라우드 리소스 고갈)이거나 DSQ 순간 가속 한도(트래픽 급증 제한)에 걸린 상태입니다. "
            f"({delay_info})"
        )
        debug_msg = (
            f"\n"
            f"================================================================================\n"
            f"🔍 [429 분석 결과: 구글 클라우드 리전 리소스 고갈 또는 트래픽 급증(Burst)]\n"
            f"================================================================================\n"
            f"• 판정 이유: 429 에러 본문에 구체적인 QuotaFailure나 quota_metric이 없이 재시도를 권고함\n"
            f"• 상세 상황: Google Cloud 특정 리전의 순간 가용 자원 고갈이거나 DSQ의 단기 트래픽 급증 제한\n"
            f"• 권장 대기시간: {delay_info}\n"
            f"• 권장 조치사항:\n"
            f"  1) 일시적인 현상이므로 지수 백오프(Exponential Backoff + Jitter)를 적용하여 잠시 후 자동 재시도\n"
            f"  2) 지속해서 발생할 경우 리전(Region)을 변경 (예: GOOGLE_CLOUD_LOCATION=us-central1, asia-northeast3 등)\n"
            f"  3) 순간적인 병렬 요청 스레드 수를 분산시켜 요청 속도 완화\n"
            f"================================================================================"
        )
        action = "잠시 대기 후 지수 백오프로 재시도하거나, 지속 발생 시 호출 리전(GOOGLE_CLOUD_LOCATION)을 다른 지역으로 변경하세요."

    return {
        "is_429": True,
        "cause_type": cause_type,
        "title": title,
        "summary": summary,
        "quota_metrics": quota_metrics,
        "retry_delay_seconds": retry_delay_seconds,
        "details_entries": details_list,
        "debug_message": debug_msg,
        "recommended_action": action,
        "raw_payload": payload,
    }


def format_quota_debug_log(exc: Exception) -> str:
    """예외 객체를 받아 즉시 콘솔에 출력하거나 로그에 남길 수 있는 디버깅 문자열을 반환합니다."""
    analysis = analyze_gemini_429_error(exc)
    return analysis["debug_message"]

