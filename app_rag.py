"""FastAPI 기반 LangChain 다중 LLM (Gemini / DeepSeek / Gemma4 / Claude Haiku) + SQLite Step-by-Step RAG 웹 서비스.

브라우저에서 직접 PDF 업로드, 청크/검색 하이퍼파라미터 변경,
Part A (인덱싱: A1~A4) 및 Part B (질의응답: B1~B4)의 각 스텝별 중간 데이터를
인터랙티브 대시보드로 실시간 확인하고 대화할 수 있는 웹 애플리케이션입니다.
클라우드 임베딩·답변(Gemini, DeepSeek, Gemma4, Claude Haiku)·OCR은 Google Cloud Vertex AI ADC를 기반으로 동작합니다.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import unicodedata
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from rag_service import RAGService
from gemini_pdf_loader import OCR_PAGE_PROMPT_TEMPLATE, OCR_SYSTEM_INSTRUCTION
from cost_calculator import (
    DEFAULT_EXCHANGE_RATE,
    get_pricing_matrix,
    simulate_all_models,
)


BASE_DIR = Path(__file__).resolve().parent
router = APIRouter()


def get_service(request: Request) -> RAGService:
    with request.app.state.service_lock:
        if request.app.state.service is None:
            request.app.state.service = RAGService()
        return request.app.state.service


# =============================================================================
# Pydantic 요청 및 응답 모델 정의
# =============================================================================
class IndexRequest(BaseModel):
    pdf_path: str = Field(..., description="인덱싱할 PDF 파일 경로")
    chunk_size: int = Field(800, ge=100, le=4000, description="청크 분할 크기")
    chunk_overlap: int = Field(150, ge=0, le=1000, description="청크 간 중첩 크기")
    embedding_model: str = Field("text-embedding-004", description="Gemini 임베딩 모델명")
    reindex: bool = Field(True, description="기존 DB 초기화 여부")
    use_embedding_cache: bool = Field(
        True, description="캐시 적중 청크의 임베딩 API 호출을 생략합니다. OCR·답변 호출은 별도입니다."
    )
    ocr_mode: Literal["auto", "gemini_ocr", "pypdf"] = Field(
        "auto",
        description="PDF 텍스트 추출 모드 ('auto': 스캔본 감지 시 Gemini 3.8 Flash OCR 자동 실행, 'gemini_ocr': 강제 OCR, 'pypdf': 일반 디지털 텍스트)",
    )
    max_pages: Optional[int] = Field(
        None, ge=1, le=1000, description="최대 로드 페이지 수 (None이면 전체)"
    )
    ocr_system_instruction: Optional[str] = Field(
        None, description="Gemini OCR 시스템 지침 (브라우저 직접 편집 가능)"
    )
    ocr_prompt_template: Optional[str] = Field(
        None, description="Gemini OCR 페이지 프롬프트 템플릿 ({page_num} 치환 지원)"
    )
    preview_page: int = Field(1, ge=1, description="미리 확인할 페이지 번호 (기본값: 1)")

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("청크 오버랩은 청크 크기보다 작아야 합니다.")
        return self


class PdfPagesRequest(BaseModel):
    pdf_path: str = Field(..., description="조회할 PDF 파일 경로")
    ocr_mode: Literal["auto", "gemini_ocr", "pypdf"] = Field(
        "auto",
        description="PDF 텍스트 추출 모드 ('auto', 'gemini_ocr', 'pypdf')",
    )
    max_pages: Optional[int] = Field(
        None, ge=1, le=1000, description="최대 로드 페이지 수 (None이면 전체)"
    )
    preview_page: int = Field(1, ge=1, description="미리 확인할 페이지 번호 (기본값: 1)")
    ocr_system_instruction: Optional[str] = Field(
        None, description="Gemini OCR 시스템 지침"
    )
    ocr_prompt_template: Optional[str] = Field(
        None, description="Gemini OCR 페이지 프롬프트 템플릿"
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="사용자 질문")
    top_k: int = Field(5, ge=1, le=50, description="검색할 유사 청크 수")
    llm_model: str = Field("gemini-3.8-flash", description="생성 모델명")
    embedding_model: str = Field("text-embedding-004", description="임베딩 모델명")
    use_embedding_cache: bool = Field(
        True, description="질문 임베딩 캐시 사용 여부"
    )


class SimulateCostRequest(BaseModel):
    input_tokens: int = Field(1000, ge=1, le=1000000, description="입력 토큰 수")
    output_tokens: int = Field(300, ge=1, le=100000, description="출력 토큰 수")
    query_count: int = Field(1, ge=1, le=100000, description="예상 질의 횟수")
    exchange_rate: float = Field(DEFAULT_EXCHANGE_RATE, gt=0, description="적용 환율 (기본 1400원/$)")


# =============================================================================
# REST API 엔드포인트
# =============================================================================
@router.get("/api/config", summary="서버 설정 및 사용 가능한 PDF 목록 조회")
def get_config(rag_service: RAGService = Depends(get_service)):
    """서버 상태, 사용 가능한 PDF 목록 및 SQLite 통계를 반환합니다."""
    pdfs = rag_service.get_available_pdfs()
    stats = rag_service.get_system_stats()
    cache_stats = rag_service.get_embedding_cache_stats()
    return {
        "status": "success",
        "available_pdfs": pdfs,
        "default_pdf": pdfs[0]["path"] if pdfs else "",
        "system_stats": stats,
        "embedding_cache_stats": cache_stats,
        "default_chunk_size": 800,
        "default_chunk_overlap": 150,
        "default_top_k": 3,
        "supported_embedding_models": ["text-embedding-004", "gemini-embedding-2"],
        "supported_llm_models": [
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
            "deepseek-ai/deepseek-v4.1-flash-maas",
            "google/gemma-4-26b-a4b-it-maas",
            "claude-haiku-5-5",
            "gemma4:e2b",
        ],
        "llm_model_details": [
            {
                "id": "gemini-3.8-flash",
                "name": "Gemini 3.8 Flash (Vertex AI)",
                "provider": "Vertex AI (ADC)",
                "type": "cloud",
            },
            {
                "id": "gemini-3.5-flash-lite",
                "name": "Gemini 3.5 Flash-Lite (Vertex AI)",
                "provider": "Vertex AI (ADC)",
                "type": "cloud",
            },
            {
                "id": "deepseek-ai/deepseek-v4.1-flash-maas",
                "name": "DeepSeek V4.1 Flash (Vertex AI)",
                "provider": "Vertex AI Model Garden",
                "type": "cloud",
            },
            {
                "id": "google/gemma-4-26b-a4b-it-maas",
                "name": "Gemma 4 26B (Vertex AI)",
                "provider": "Vertex AI Model Garden",
                "type": "cloud",
            },
            {
                "id": "claude-haiku-5-5",
                "name": "Claude Haiku 5.5 (Vertex AI)",
                "provider": "Anthropic on Vertex AI",
                "type": "cloud",
            },
            {
                "id": "gemma4:e2b",
                "name": "Gemma 4 E2B (Ollama 로컬)",
                "provider": "Ollama (로컬)",
                "type": "local",
            },
        ],
        "supported_ocr_modes": [
            {"value": "auto", "label": "자동 감지 (스캔본/텍스트 누락 시 Gemini 3.8 Flash OCR 실행)"},
            {"value": "gemini_ocr", "label": "항상 Gemini 3.8 Flash OCR (이미지 묘사 제외, 텍스트만 전사)"},
            {"value": "pypdf", "label": "일반 디지털 텍스트 추출 (PyPDF)"},
        ],
        "default_ocr_mode": "auto",
        "default_ocr_system_instruction": OCR_SYSTEM_INSTRUCTION,
        "default_ocr_prompt_template": OCR_PAGE_PROMPT_TEMPLATE,
        "cost_stats": rag_service.cost_tracker.get_stats(),
    }


@router.get("/api/cost/pricing", summary="공식 요율표 전체 조회")
def get_cost_pricing():
    """Vertex AI Model Garden 및 Partner Models 공식 요율표를 반환합니다."""
    return {
        "status": "success",
        "exchange_rate": DEFAULT_EXCHANGE_RATE,
        "pricing_matrix": get_pricing_matrix(),
        "sources": [
            {
                "title": "Google Cloud Model Garden First-Party & Open-Weights MaaS 요율",
                "url": "https://cloud.google.com/model-garden/pricing#cost-of-building-and-deploying-ai-models-in-agent-platform",
            },
            {
                "title": "Google Cloud Model Garden Partner Models (Claude) 요율",
                "url": "https://cloud.google.com/model-garden/partner-models/pricing",
            },
        ],
    }


@router.get("/api/cost/stats", summary="세션 누적 토큰 및 과금 현황 통계 조회")
def get_cost_stats(rag_service: RAGService = Depends(get_service)):
    """현재 세션에서 발생한 모델별 누적 토큰 수, 과금액(USD/KRW) 및 질의 이력을 반환합니다."""
    return {
        "status": "success",
        **rag_service.cost_tracker.get_stats(),
    }


@router.post("/api/cost/reset", summary="세션 누적 과금 통계 초기화")
def reset_cost_stats(rag_service: RAGService = Depends(get_service)):
    """세션 누적 과금 및 토큰 통계를 초기화합니다."""
    rag_service.cost_tracker.reset()
    return {"status": "success", "message": "누적 과금 및 토큰 통계가 성공적으로 초기화되었습니다."}


@router.post("/api/cost/simulate", summary="모델별 비용 비교 시뮬레이션")
def simulate_cost(req: SimulateCostRequest):
    """주어진 입력/출력 토큰 및 질의 횟수에 대한 모든 지원 모델의 예상 비용을 비교합니다."""
    results = simulate_all_models(
        input_tokens=req.input_tokens,
        output_tokens=req.output_tokens,
        query_count=req.query_count,
        exchange_rate=req.exchange_rate,
    )
    return {
        "status": "success",
        "input_tokens": req.input_tokens,
        "output_tokens": req.output_tokens,
        "query_count": req.query_count,
        "exchange_rate": req.exchange_rate,
        "comparisons": results,
    }


@router.get("/api/cache/embedding/stats", summary="임베딩 캐시 현황 통계 조회")
def get_embedding_cache_stats(rag_service: RAGService = Depends(get_service)):
    """SQLite 임베딩 캐시의 총 저장 수, 적중 횟수, 모델별 통계를 반환합니다."""
    return rag_service.get_embedding_cache_stats()


@router.post("/api/cache/embedding/clear", summary="임베딩 캐시 초기화")
def clear_embedding_cache(model_name: Optional[str] = Query(None, description="특정 모델의 캐시만 삭제할 경우 지정"), rag_service: RAGService = Depends(get_service)):
    """임베딩 캐시를 초기화합니다."""
    deleted = rag_service.clear_embedding_cache(model_name=model_name)
    return {
        "status": "success",
        "deleted_entries": deleted,
        "message": f"{deleted}개의 임베딩 캐시 항목이 성공적으로 삭제되었습니다."
    }


@router.post("/api/upload", summary="PDF 업로드 (100MiB 이하)")
async def upload_pdf(file: UploadFile = File(...), rag_service: RAGService = Depends(get_service)):
    name = unicodedata.normalize("NFC", (file.filename or "").replace("\\", "/").split("/")[-1])
    name = re.sub(r"[^\w .()-]", "_", name)
    if not name.lower().endswith(".pdf") or len(name) > 180 or name.startswith("."):
        raise HTTPException(400, "올바른 PDF 파일명을 사용하세요.")
    temporary = None
    try:
        with rag_service._operation_lock:
            with tempfile.NamedTemporaryFile(dir=rag_service.upload_dir, suffix=".tmp", delete=False) as buffer:
                temporary = Path(buffer.name)
                size = 0
                while chunk := await file.read(1024 * 1024):
                    if size == 0 and not chunk.startswith(b"%PDF-"):
                        raise HTTPException(400, "유효한 PDF 파일을 선택하세요.")
                    size += len(chunk)
                    if size > 100 * 1024 * 1024:
                        raise HTTPException(413, "100MiB 이하의 PDF를 선택하세요.")
                    buffer.write(chunk)
            if not size:
                raise HTTPException(400, "빈 파일은 업로드할 수 없습니다.")
            import pypdf
            if len(pypdf.PdfReader(temporary).pages) > 1000:
                raise HTTPException(400, "1000쪽 이하의 PDF를 선택하세요.")
            destination = rag_service.upload_dir / name
            # Never follow an existing symlink at the destination.
            if destination.is_symlink():
                raise HTTPException(400, "이 파일명으로 저장할 수 없습니다.")
            temporary.replace(destination)
        return {"status":"success", "file_name":name, "file_path":str(destination), "size_kb":round(size/1024,1)}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "PDF 형식을 확인할 수 없습니다.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        await file.close()


@router.post("/api/pdf/pages", summary="PDF 페이지 목록 및 본문 조회 (인덱싱 없이 바로 확인)")
def get_pdf_pages(req: PdfPagesRequest, rag_service: RAGService = Depends(get_service)):
    """지정한 PDF 파일의 페이지 목록과 텍스트를 추출하여 반환합니다."""
    try:
        result = rag_service.get_pdf_pages(
            pdf_path=req.pdf_path,
            ocr_mode=req.ocr_mode,
            max_pages=req.max_pages,
            ocr_system_instruction=req.ocr_system_instruction,
            ocr_prompt_template=req.ocr_prompt_template,
            preview_page=req.preview_page,
        )
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="처리하지 못했습니다. ADC·프로젝트·모델 접근 권한과 네트워크를 확인하세요.")


@router.post("/api/index", summary="Part A 인덱싱 실행 (A1~A4 스텝별 결과 반환)")
def run_indexing(req: IndexRequest, rag_service: RAGService = Depends(get_service)):
    """지정한 PDF 파일을 읽고 분할, 임베딩 후 SQLite에 저장하며 각 스텝별 결과를 반환합니다."""
    try:
        result = rag_service.run_indexing(
            pdf_path=req.pdf_path,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            embedding_model=req.embedding_model,
            reindex=req.reindex,
            ocr_mode=req.ocr_mode,
            max_pages=req.max_pages,
            ocr_system_instruction=req.ocr_system_instruction,
            ocr_prompt_template=req.ocr_prompt_template,
            preview_page=req.preview_page,
            use_embedding_cache=req.use_embedding_cache,
        )
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="처리하지 못했습니다. ADC·프로젝트·모델 접근 권한과 네트워크를 확인하세요.")


@router.post("/api/index/stream", summary="Part A 인덱싱 실시간 진행률 스트리밍 (NDJSON)")
def run_indexing_stream(req: IndexRequest, rag_service: RAGService = Depends(get_service)):
    """지정한 PDF 인덱싱을 실행하며 실시간 진행율(퍼센트, 스텝, 상태 메시지)과 최종 데이터를 스트리밍합니다."""
    event_queue: queue.Queue = queue.Queue()

    def run_worker():
        try:
            def on_progress(event: Dict[str, Any]):
                event_queue.put({"type": "progress", **event})

            result = rag_service.run_indexing(
                pdf_path=req.pdf_path,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
                embedding_model=req.embedding_model,
                reindex=req.reindex,
                ocr_mode=req.ocr_mode,
                max_pages=req.max_pages,
                ocr_system_instruction=req.ocr_system_instruction,
                ocr_prompt_template=req.ocr_prompt_template,
                preview_page=req.preview_page,
                progress_callback=on_progress,
                use_embedding_cache=req.use_embedding_cache,
            )
            event_queue.put({"type": "result", "data": result})
        except Exception as exc:
            event_queue.put({"type": "error", "message": str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else "인덱싱 실패: ADC·모델·네트워크를 확인하세요. 기존 인덱스는 유지됩니다."})
        finally:
            event_queue.put(None)

    threading.Thread(target=run_worker, daemon=True).start()

    def event_stream():
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/query", summary="Part B 질의응답 실행 (B1~B4 스텝별 결과 반환)")
def run_query(req: QueryRequest, rag_service: RAGService = Depends(get_service)):
    """사용자의 질문을 임베딩하고 SQLite에서 관련 청크 검색 및 프롬프트 조립 후 Gemini로 답변을 생성합니다."""
    try:
        result = rag_service.run_query(
            question=req.question,
            top_k=req.top_k,
            llm_model=req.llm_model,
            embedding_model=req.embedding_model,
            use_embedding_cache=req.use_embedding_cache,
        )
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="처리하지 못했습니다. ADC·프로젝트·모델 접근 권한과 네트워크를 확인하세요.")


@router.get("/api/db/sample", summary="SQLite에 저장된 레코드 1건 직접 확인")
def get_db_sample(rag_service: RAGService = Depends(get_service)):
    """SQLite DB에서 실제 저장된 데이터 1건을 조회하여 반환합니다."""
    rec = rag_service.vector_store.get_sample_record()
    stats = rag_service.vector_store.get_stats()
    return {
        "stats": stats,
        "sample_record": rec,
    }


# =============================================================================
# RAG 실험 및 벤치마크 분석 API (최적 조합 분석용)
# =============================================================================
@router.get("/api/experiments/logs", summary="RAG 인덱싱/질의 실험 로그 목록 조회")
def get_experiment_logs(limit: int = Query(50, ge=1, le=500), event_type: Optional[str] = None, rag_service: RAGService = Depends(get_service)):
    """최근 인덱싱 및 질의 실험 로그 목록을 반환합니다 (필터: indexing, query)."""
    logs = rag_service.experiment_logger.get_recent_logs(limit=limit, event_type=event_type)
    return {
        "status": "success",
        "count": len(logs),
        "logs": logs,
    }


@router.get("/api/experiments/summary", summary="RAG 최적 조합 분석 및 성능 요약 통계")
def get_experiment_summary(rag_service: RAGService = Depends(get_service)):
    """모델별, 청크 크기별, 캐시 유무별 성능 및 비용 통계와 최적 조합 추천을 반환합니다."""
    summary = rag_service.experiment_logger.get_analysis_summary()
    return {
        "status": "success",
        **summary,
    }


@router.get("/api/experiments/export/csv", response_class=PlainTextResponse, summary="실험 로그 CSV 내보내기")
def export_experiment_csv(event_type: Optional[str] = None, rag_service: RAGService = Depends(get_service)):
    """실험 로그를 CSV 포맷으로 내보냅니다."""
    csv_data = rag_service.experiment_logger.export_csv(event_type=event_type)
    headers = {
        "Content-Disposition": f"attachment; filename=rag_experiments_{int(time.time())}.csv",
        "Content-Type": "text/csv; charset=utf-8",
    }
    return PlainTextResponse(content=csv_data, headers=headers)


@router.get("/api/experiments/export/jsonl", response_class=FileResponse, summary="실험 로그 JSONL 파일 다운로드")
def export_experiment_jsonl(rag_service: RAGService = Depends(get_service)):
    """실험 로그 JSONL 파일을 직접 다운로드합니다."""
    if not rag_service.experiment_logger.jsonl_path.exists():
        raise HTTPException(status_code=404, detail="실험 로그 파일이 존재하지 않습니다.")
    headers = {
        "Content-Disposition": f"attachment; filename=rag_experiments_{int(time.time())}.jsonl",
    }
    return FileResponse(str(rag_service.experiment_logger.jsonl_path), media_type="application/x-ndjson", headers=headers)


# =============================================================================
# 웹 UI 정적 파일 서빙
# =============================================================================
STATIC_DIR = BASE_DIR / "static"

@router.get("/", response_class=FileResponse, summary="RAG 대시보드 웹 UI")
def serve_dashboard():
    """RAG 시각화 대시보드 웹 페이지를 서빙합니다."""
    ui_path = STATIC_DIR / "rag_dashboard.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="rag_dashboard.html 파일을 찾을 수 없습니다.")
    return FileResponse(ui_path)


def create_app(service: Optional[RAGService] = None) -> FastAPI:
    application = FastAPI(title="RAG Step-by-Step Student Studio", version="1.2.0")
    application.state.service = service
    application.state.service_lock = threading.Lock()
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @application.middleware("http")
    async def local_origin_only(request: Request, call_next):
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if origin and origin != expected:
            return JSONResponse({"detail":"다른 웹사이트에서 이 로컬 서비스에 접근할 수 없습니다."}, status_code=403)
        return await call_next(request)

    application.include_router(router)
    return application


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"\n🚀 LangChain RAG Web Service 시작: http://127.0.0.1:{port}")
    uvicorn.run("app_rag:app", host="127.0.0.1", port=port, reload=False)

