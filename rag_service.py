"""LangChain + Gemini 3.8 Flash + SQLite RAG 비즈니스 로직 서비스 모듈.

FastAPI 웹 서비스 및 CLI에서 공통으로 재사용할 수 있도록
A단계(미리 준비하기/인덱싱)와 B단계(질문에 답하기/런타임)를 모듈화하여
각 스텝별 상세 실행 결과(중간 데이터)를 구조화된 딕셔너리로 반환합니다.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", category=FutureWarning)

import google.auth
import google.auth.transport.requests
import numpy as np
from dotenv import load_dotenv
import pypdf

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sqlite_vector_store import SQLiteVectorStore
from gemini_pdf_loader import GeminiFlashPDFLoader
from gemini_quota_debugger import analyze_gemini_429_error, format_quota_debug_log
from embedding_cache import EmbeddingCache, CachedEmbeddings
from cost_calculator import (
    calculate_cost,
    calculate_embedding_cost,
    extract_tokens_from_response,
    CostTracker,
)
from rag_experiment_logger import RAGExperimentLogger
from runtime_config import data_directory

BASE_DIR = Path(__file__).resolve().parent

# 배포 폴더의 사용자 설정만 읽습니다. 상위 폴더의 개인 설정은 읽지 않습니다.
load_dotenv(BASE_DIR / ".env")


def get_gcp_access_token() -> str:
    """Google Cloud Application Default Credentials(ADC)로부터 Access Token을 발급받습니다."""
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    return credentials.token


def extract_llm_text(response: Any) -> str:
    """ChatGoogleGenerativeAI, ChatOpenAI, ChatAnthropicVertex 응답 객체에서 텍스트 본문만 추출합니다."""
    if hasattr(response, "text") and response.text:
        return response.text
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif "text" in part:
                    texts.append(part["text"])
        return "".join(texts)
    return str(content)


class RAGService:
    """RAG 파이프라인 전 단계를 총괄하는 서비스 클래스."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        default_embedding_model: str = "text-embedding-004",
        default_llm_model: str = "gemini-3.8-flash",
        data_dir: Optional[str | Path] = None,
        allowed_pdf_dirs: Optional[List[Path]] = None,
    ):
        self.data_dir = Path(data_dir).resolve() if data_dir else (Path(db_path).resolve().parent if db_path else data_directory())
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        self.allowed_pdf_dirs = [p.resolve() for p in (allowed_pdf_dirs or [BASE_DIR / "pdf", self.upload_dir])]
        self.db_path = str(Path(db_path).resolve() if db_path else self.data_dir / "rag_sqlite.db")
        self._operation_lock = threading.RLock()
        self.cost_tracker = CostTracker()
        self.experiment_logger = RAGExperimentLogger(self.data_dir / "logs" / "rag_experiments.jsonl", Path(self.db_path))
        self.default_embedding_model = default_embedding_model
        self.default_llm_model = default_llm_model

        # 임베딩 캐시 및 모델 인스턴스 관리
        self._embedding_cache = EmbeddingCache(cache_dir=self.data_dir / "cache")
        self._embedders: Dict[str, Embeddings] = {}
        self._llms: Dict[str, BaseChatModel] = {}
        self.active_auth_method: Optional[str] = None

        # 기본 벡터 스토어 초기화
        self.vector_store = SQLiteVectorStore(
            db_path=self.db_path,
            embedding_function=None,
        )

    @property
    def embedding_cache(self) -> EmbeddingCache:
        """지연 초기화를 지원하는 임베딩 캐시 인스턴스."""
        if not hasattr(self, "_embedding_cache") or self._embedding_cache is None:
            self._embedding_cache = EmbeddingCache(cache_dir=self.data_dir / "cache")
        return self._embedding_cache

    @embedding_cache.setter
    def embedding_cache(self, cache: EmbeddingCache) -> None:
        self._embedding_cache = cache

    # -------------------------------------------------------------------------
    # 인증 및 모델 인스턴스 관리
    # -------------------------------------------------------------------------
    def get_embedder(
        self, model_name: Optional[str] = None, use_cache: bool = True
    ) -> Embeddings:
        name = model_name or self.default_embedding_model
        if name in self._embedders:
            cached_emb = self._embedders[name]
            if not use_cache and isinstance(cached_emb, CachedEmbeddings):
                return cached_emb.underlying_embeddings
            return cached_emb

        # 클라우드 임베딩은 ADC만 사용하며, 초기화/호출 오류를 그대로 전달합니다.
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
        if not project or project == "YOUR_PROJECT_ID":
            raise ValueError(
                "Google Cloud 프로젝트 ID를 찾을 수 없습니다. "
                "GOOGLE_CLOUD_PROJECT를 설정하거나 ADC 프로젝트 설정을 확인하세요."
            )
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

        emb = GoogleGenerativeAIEmbeddings(
            model=name,
            vertexai=True,
            credentials=credentials,
            project=project,
            location=location,
            api_key=None,
        )
        cached_emb = CachedEmbeddings(
            underlying_embeddings=emb,
            cache=self.embedding_cache,
            model_name=name,
        )
        self._embedders[name] = cached_emb
        self.active_auth_method = f"Vertex AI ADC (Project: {project}, Region: {location})"
        return cached_emb if use_cache else emb

    def get_embedding_cache_stats(self) -> Dict[str, Any]:
        """임베딩 캐시 저장소 현황 및 통계를 조회합니다."""
        return self.embedding_cache.get_stats()

    def clear_embedding_cache(self, model_name: Optional[str] = None) -> int:
        """임베딩 캐시를 초기화합니다."""
        return self.embedding_cache.clear(model_name=model_name)

    def get_llm(self, model_name: Optional[str] = None) -> BaseChatModel:
        raw_name = model_name or self.default_llm_model
        name = raw_name.strip()
        lower_name = name.lower()

        # 1. Ollama 로컬 모델 분기 (: 가 포함되어 있거나 ollama 명시된 경우)
        if ":" in name or "ollama" in lower_name:
            if name in self._llms:
                return self._llms[name]
            try:
                llm = ChatOllama(
                    model=name,
                    temperature=0.2,
                )
                self._llms[name] = llm
                return llm
            except Exception as e:
                raise RuntimeError(
                    f"Ollama 모델 '{name}' 초기화에 실패했습니다: {e}. Ollama가 실행 중이고 해당 모델이 설치되어 있는지 확인하세요."
                )

        # Google Cloud 공통 인증 정보
        credentials, default_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
        if not project or project == "YOUR_PROJECT_ID":
            raise ValueError(
                "Google Cloud 프로젝트 ID를 찾을 수 없습니다. "
                "GOOGLE_CLOUD_PROJECT를 설정하거나 ADC 프로젝트 설정을 확인하세요."
            )
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

        # 2. Claude Haiku 5.5 (Anthropic on Vertex AI)
        if any(k in lower_name for k in ("claude", "haiku")):
            model_id = os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-5-5") if lower_name in ("haiku", "haiku-5.5", "haiku 5.5", "claude-haiku", "claude-haiku-5-5") else name
            if model_id in self._llms:
                return self._llms[model_id]
            llm = ChatAnthropicVertex(
                model_name=model_id,
                project=project,
                location=location,
                max_output_tokens=4096,
                streaming=False,
            )
            self._llms[model_id] = llm
            return llm

        # 3. DeepSeek V4.1 Flash (Vertex AI Model Garden - OpenAI 호환 MaaS)
        if "deepseek" in lower_name:
            model_id = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/deepseek-v4.1-flash-maas") if lower_name in ("deepseek", "deepseek-v4.1-flash", "deepseek-ai/deepseek-v4.1-flash-maas") else name
            endpoint_host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            base_url = f"https://{endpoint_host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
            if model_id in self._llms:
                return self._llms[model_id]
            llm = ChatOpenAI(
                model=model_id,
                api_key=get_gcp_access_token,
                openai_api_base=base_url,
                temperature=0.7,
                streaming=False,
            )
            self._llms[model_id] = llm
            return llm

        # 4. Gemma 4 26B (Vertex AI Model Garden - OpenAI 호환 MaaS)
        if "gemma" in lower_name:
            model_id = os.getenv("GEMMA_MODEL", "google/gemma-4-26b-a4b-it-maas") if lower_name in ("gemma4", "gemma-4", "gemma", "google/gemma-4-26b-a4b-it-maas") else name
            endpoint_host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
            base_url = f"https://{endpoint_host}/v1/projects/{project}/locations/{location}/endpoints/openapi"
            if model_id in self._llms:
                return self._llms[model_id]
            llm = ChatOpenAI(
                model=model_id,
                api_key=get_gcp_access_token,
                openai_api_base=base_url,
                temperature=0.7,
                streaming=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
            self._llms[model_id] = llm
            return llm

        # 5. Gemini 3.5 Flash-Lite (Vertex AI ADC)
        if "flash-lite" in lower_name or "lite" in lower_name:
            model_id = os.getenv("GEMINI_FLASH_LITE_MODEL", "gemini-3.5-flash-lite") if lower_name in ("gemini-3.5-flash-lite", "flash-lite", "gemini-flash-lite") else name
            if model_id in self._llms:
                return self._llms[model_id]
            llm = ChatGoogleGenerativeAI(
                model=model_id,
                vertexai=True,
                credentials=credentials,
                project=project,
                location=location,
                api_key=None,
                streaming=False,
                timeout=120,
                max_retries=1,
            )
            self._llms[model_id] = llm
            return llm

        # 6. 기본 Gemini 모델 (gemini-3.8-flash 등)
        model_id = name
        if model_id in self._llms:
            return self._llms[model_id]
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            vertexai=True,
            credentials=credentials,
            project=project,
            location=location,
            api_key=None,
            streaming=False,
        )
        self._llms[model_id] = llm
        return llm

    # -------------------------------------------------------------------------
    # 시스템 탐색 및 상태 조회
    # -------------------------------------------------------------------------
    def get_available_pdfs(self) -> List[Dict[str, Any]]:
        """서버 상에서 사용 가능한 PDF 목록을 반환합니다."""
        pdf_paths: List[Path] = []
        target_dirs = self.allowed_pdf_dirs
        for d in target_dirs:
            if d.exists() and d.is_dir():
                pdf_paths.extend(d.glob("*.pdf"))

        unique_files: Dict[str, Path] = {}
        for p in pdf_paths:
            try:
                resolved = p.resolve()
                if resolved.is_file() and resolved.name not in unique_files:
                    unique_files[resolved.name] = resolved
            except Exception:
                continue

        result = []
        for name, p in unique_files.items():
            result.append({
                "name": name,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
        return sorted(result, key=lambda x: x["name"])

    def get_system_stats(self) -> Dict[str, Any]:
        """SQLite DB 상태 및 모델 정보 조회."""
        db_stats = self.vector_store.get_stats()
        sample_rec = self.vector_store.get_sample_record()
        return {
            "db_stats": db_stats,
            "sample_record": sample_rec,
            "default_embedding_model": self.default_embedding_model,
            "default_llm_model": self.default_llm_model,
            "active_auth_method": self.active_auth_method or "감지 대기 중",
        }

    # -------------------------------------------------------------------------
    # PDF 텍스트/OCR 추출 및 페이지 열람
    # -------------------------------------------------------------------------
    def resolve_pdf(self, pdf_path: str) -> Path:
        path = Path(pdf_path).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        path = path.resolve()
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ValueError("허용된 폴더 안의 PDF 파일을 선택하세요.")
        if not any(path.is_relative_to(folder) for folder in self.allowed_pdf_dirs):
            raise ValueError("지정된 PDF/업로드 폴더 밖의 파일은 읽을 수 없습니다.")
        if path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("100MiB 이하의 PDF를 선택하세요.")
        return path

    def _extract_documents(
        self, path: Path, ocr_mode: str = "auto", max_pages: Optional[int] = None,
        ocr_system_instruction: Optional[str] = None, ocr_prompt_template: Optional[str] = None,
        page_progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        if ocr_mode not in {"auto", "gemini_ocr", "pypdf"}:
            raise ValueError("지원하지 않는 OCR 모드입니다.")
        if max_pages is not None and max_pages < 1:
            raise ValueError("최대 페이지 수는 1 이상이어야 합니다.")
        reader = pypdf.PdfReader(str(path))
        total_pages = len(reader.pages)
        if total_pages > 1000:
            raise ValueError("실습에서는 1000쪽 이하의 PDF만 처리합니다.")
        count = min(max_pages, total_pages) if max_pages else total_pages
        by_page, page_stats, needs_ocr = {}, [], []
        for index in range(count):
            text = reader.pages[index].extract_text() or ""
            by_page[index + 1] = Document(page_content=text, metadata={"source":path.name,"page":index+1,"path":str(path)})
            page_stats.append({"page":index+1,"char_count":len(text.strip()),"status":"텍스트" if text.strip() else "빈 페이지/스캔본"})
            if ocr_mode == "gemini_ocr" or (ocr_mode == "auto" and len(text.strip()) < 50):
                needs_ocr.append(index)
            if page_progress_callback:
                page_progress_callback(index+1, count, f"{len(text.strip())}자 추출")
        if needs_ocr:
            loader = GeminiFlashPDFLoader(
                path, page_indices=needs_ocr, cache_dir=self.data_dir / "ocr",
                progress_callback=page_progress_callback, system_instruction=ocr_system_instruction,
                prompt_template=ocr_prompt_template, usage_callback=self.cost_tracker.record_ocr_job,
            )
            for document in loader.load():
                by_page[document.metadata["page"]] = document
        documents = [by_page[n] for n in sorted(by_page)]
        if not any(d.page_content.strip() for d in documents):
            raise ValueError("PDF에서 추출된 텍스트가 없습니다. OCR 설정과 원본을 확인하세요.")
        used = bool(needs_ocr)
        return documents, total_pages, page_stats, ("PyPDF + Gemini OCR" if used else "PyPDFLoader"), used, (f"{len(needs_ocr)}쪽을 페이지 단위로 OCR 처리했습니다." if used else "")

    def get_pdf_pages(self, *args, **kwargs):
        with self._operation_lock:
            return self._get_pdf_pages(*args, **kwargs)

    def _get_pdf_pages(
        self,
        pdf_path: str,
        ocr_mode: str = "auto",
        max_pages: Optional[int] = None,
        ocr_system_instruction: Optional[str] = None,
        ocr_prompt_template: Optional[str] = None,
        preview_page: int = 1,
    ) -> Dict[str, Any]:
        """PDF 파일의 페이지별 전체 텍스트를 추출하여 반환합니다 (인덱싱 없이도 페이지 열람 및 탐색 가능)."""
        path = self.resolve_pdf(pdf_path)

        documents, total_pages, page_stats, loader_type, ocr_used, ocr_reason = (
            self._extract_documents(
                path,
                ocr_mode=ocr_mode,
                max_pages=max_pages,
                ocr_system_instruction=ocr_system_instruction,
                ocr_prompt_template=ocr_prompt_template,
            )
        )

        pages_data = [
            {
                "page": d.metadata.get("page", idx + 1),
                "char_count": len(d.page_content.strip()),
                "content": d.page_content.strip(),
                "ocr_model": d.metadata.get("ocr_model", ""),
                "cached": d.metadata.get("cached", False),
            }
            for idx, d in enumerate(documents)
        ]

        preview_doc = None
        for d in documents:
            if d.metadata.get("page") == preview_page:
                preview_doc = d
                break
        if not preview_doc and documents:
            preview_doc = documents[0]

        return {
            "status": "success",
            "file_name": path.name,
            "file_path": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "total_pages": total_pages,
            "extracted_pages": len(pages_data),
            "loader_type": loader_type,
            "ocr_used": ocr_used,
            "ocr_reason": ocr_reason,
            "selected_page": preview_doc.metadata.get("page", 1) if preview_doc else 1,
            "preview_text": preview_doc.page_content.strip() if preview_doc else "",
            "pages": pages_data,
        }

    # -------------------------------------------------------------------------
    # PART A. 미리 준비하기 (인덱싱 파이프라인)
    # -------------------------------------------------------------------------
    def run_indexing(self, *args, **kwargs):
        with self._operation_lock:
            return self._run_indexing(*args, **kwargs)

    def _run_indexing(
        self,
        pdf_path: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedding_model: Optional[str] = None,
        reindex: bool = True,
        ocr_mode: str = "auto",
        max_pages: Optional[int] = None,
        ocr_system_instruction: Optional[str] = None,
        ocr_prompt_template: Optional[str] = None,
        preview_page: int = 1,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        use_embedding_cache: bool = True,
    ) -> Dict[str, Any]:
        """A1~A4 단계를 순차 실행하고 각 단계별 중간 출력 데이터를 딕셔너리로 반환합니다.

        ocr_mode:
            - 'auto': 일반 텍스트 추출 시도 후 텍스트가 없으면(스캔본/이미지) Gemini 3.8 Flash OCR 자동 실행 (추천)
            - 'gemini_ocr': 항상 Gemini 3.8 Flash OCR 실행 (이미지 묘사 제외, 텍스트만 추출)
            - 'pypdf': 일반 디지털 텍스트 추출만 사용
        use_embedding_cache:
            - True: SQLite 기반 임베딩 캐시 사용 (기존 청크 벡터 0초 재사용, API 비용 절감)
        """
        def report(percent: int, step: str, message: str, detail: str = ""):
            if progress_callback:
                try:
                    progress_callback({
                        "percent": max(0, min(100, int(percent))),
                        "step": step,
                        "message": message,
                        "detail": detail,
                    })
                except Exception:
                    pass

        if chunk_overlap >= chunk_size or chunk_overlap < 0 or chunk_size < 100:
            raise ValueError("청크 오버랩은 청크 크기보다 작아야 합니다.")
        pipeline_start_t = time.time()
        report(5, "INIT", "인덱싱 준비 및 임베딩 모델 연결 확인...", "초기화")

        path = self.resolve_pdf(pdf_path)
        model_name = embedding_model or self.default_embedding_model
        if not reindex:
            self.vector_store.assert_model(model_name)
        embedder = self.get_embedder(model_name, use_cache=use_embedding_cache)

        # =====================================================================
        # Step A1. 문서 읽기 (PDF -> 텍스트 / Gemini 3.8 Flash OCR)
        # =====================================================================
        a1_start_t = time.time()
        report(10, "A1", f"Step A1: PDF 문서 로드 시작 ({path.name})...", "문서 로드")

        def on_page_progress(cur_page: int, tot_pages: int, info: str):
            tot = max(1, tot_pages)
            pct = 10 + int(25 * (cur_page / tot))
            report(pct, "A1", f"Step A1: 문서 텍스트 추출 중 ({cur_page}/{tot}쪽)", info)

        documents, total_pages, page_stats, loader_type, ocr_used, ocr_reason = (
            self._extract_documents(
                path,
                ocr_mode=ocr_mode,
                max_pages=max_pages,
                ocr_system_instruction=ocr_system_instruction,
                ocr_prompt_template=ocr_prompt_template,
                page_progress_callback=on_page_progress,
            )
        )
        report(35, "A1", f"Step A1 완료: {len(documents)}개 페이지 추출 완료", f"로더: {loader_type}")

        preview_doc = None
        for d in documents:
            if d.metadata.get("page") == preview_page:
                preview_doc = d
                break
        if not preview_doc and documents:
            preview_doc = documents[0]

        preview_text = (
            preview_doc.page_content.strip()
            if preview_doc
            else ""
        )

        pages_data = [
            {
                "page": d.metadata.get("page", idx + 1),
                "char_count": len(d.page_content.strip()),
                "content": d.page_content.strip(),
                "ocr_model": d.metadata.get("ocr_model", ""),
                "cached": d.metadata.get("cached", False),
            }
            for idx, d in enumerate(documents)
        ]

        step_a1_data = {
            "step": "A1",
            "title": "문서 읽기 (PDF → 텍스트 / Gemini OCR)",
            "file_name": path.name,
            "file_path": str(path),
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "total_pages": total_pages,
            "extracted_pages": len(documents),
            "loader_type": loader_type,
            "ocr_used": ocr_used,
            "ocr_reason": ocr_reason,
            "page_stats": page_stats[:10],
            "selected_page": preview_doc.metadata.get("page", 1) if preview_doc else 1,
            "preview_text": preview_text,
            "pages": pages_data,
        }
        a1_elapsed = round(time.time() - a1_start_t, 3)

        # =====================================================================
        # Step A2. 청크로 나누기 (작은 원문 조각) & 중복률 분석 (Chunk Hash & Inclusion)
        # =====================================================================
        a2_start_t = time.time()
        report(40, "A2", "Step A2: 텍스트 청크 분할 및 중복 검사 중...", f"크기 {chunk_size}자 / 중첩 {chunk_overlap}자")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)

        # 기존 DB에 이미 저장되어 있는 청크의 content_hash 집합 조회
        existing_hashes = self.vector_store.get_existing_content_hashes()

        # 청크별 결정론적 content_hash 및 고유 chunk_id 부여
        duplicate_chunk_count = 0
        matched_sources = set()

        source_id = hashlib.sha256(unicodedata.normalize("NFC", path.name).casefold().encode("utf-8")).hexdigest()
        page_hashes = {d.metadata["page"]: hashlib.sha256(d.page_content.encode("utf-8")).hexdigest() for d in documents}
        page_offsets = {}
        for idx, chunk in enumerate(chunks):
            # 내용 텍스트 기준 SHA-256 (앞 16자리)
            c_content_clean = " ".join(chunk.page_content.split())
            c_hash = hashlib.sha256(c_content_clean.encode("utf-8")).hexdigest()[:16]
            
            chunk.metadata["content_hash"] = c_hash
            chunk.metadata["file_name"] = path.name
            chunk.metadata["source"] = str(path)
            # 결정론적 고유 ID (파일명_해시)
            page = chunk.metadata["page"]
            offset = page_offsets.get(page, 0)
            page_offsets[page] = offset + 1
            chunk.metadata["source_id"] = source_id
            chunk.metadata["embedding_model"] = model_name
            chunk.metadata["chunk_id"] = hashlib.sha256(f"{source_id}:{page}:{offset}:{c_hash}".encode()).hexdigest()

            is_dup = c_hash in existing_hashes
            chunk.metadata["is_duplicate"] = is_dup
            if is_dup:
                duplicate_chunk_count += 1
                matched_sources.add(existing_hashes[c_hash])

        total_chunks_cnt = len(chunks)
        dup_ratio = round((duplicate_chunk_count / total_chunks_cnt * 100), 1) if total_chunks_cnt > 0 else 0.0

        chunk_lens = [len(c.page_content) for c in chunks]
        all_chunks_data = [
            {
                "index": idx + 1,
                "chunk_id": c.metadata["chunk_id"],
                "content_hash": c.metadata.get("content_hash"),
                "is_duplicate": c.metadata.get("is_duplicate", False),
                "page": c.metadata.get("page", 1),
                "length": len(c.page_content),
                "content": c.page_content.strip(),
            }
            for idx, c in enumerate(chunks)
        ]

        # 중복 분석 결과 진단
        dup_diagnosis = "NEW"  # NEW, SUBSET, PARTIAL_DUP
        if dup_ratio >= 90.0:
            dup_diagnosis = "SUBSET"  # 기존 문서에 완전히 포함된 부분집합/발췌본
        elif dup_ratio > 0.0:
            dup_diagnosis = "PARTIAL_DUP"

        step_a2_data = {
            "step": "A2",
            "title": "청크로 나누기 (작은 원문 조각)",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_chunks": total_chunks_cnt,
            "min_length": min(chunk_lens) if chunk_lens else 0,
            "avg_length": round(sum(chunk_lens) / len(chunk_lens), 1) if chunk_lens else 0,
            "max_length": max(chunk_lens) if chunk_lens else 0,
            "sample_chunks": all_chunks_data[:3],
            "chunks": all_chunks_data,
            # 중복 검사 결과
            "duplicate_chunk_count": duplicate_chunk_count,
            "duplicate_ratio": dup_ratio,
            "duplicate_diagnosis": dup_diagnosis,
            "matched_existing_sources": sorted(list(matched_sources))[:5],
        }
        report(
            50,
            "A2",
            f"Step A2 완료: 총 {total_chunks_cnt}개 청크 (중복률 {dup_ratio}%)",
            f"진단: {dup_diagnosis} (기존 일치 {duplicate_chunk_count}개)"
        )

        # Preserve provenance for every source/page; cached embeddings avoid duplicate API work.
        chunks_to_embed = chunks
        a2_elapsed = round(time.time() - a2_start_t, 3)

        # =====================================================================
        # Step A3. 임베딩하기 (청크 -> 숫자 배열)
        # =====================================================================
        report(55, "A3", f"Step A3: {model_name} 임베딩 벡터 생성 시작...", f"총 {len(chunks_to_embed)}개 청크")
        texts = [c.page_content for c in chunks_to_embed]
        start_t = time.time()
        embeddings: List[List[float]] = []
        batch_size = 16
        total_chunks = len(texts)

        # 사전 캐시 적중 여부 파악
        cache_hits = 0
        cache_misses = total_chunks
        if use_embedding_cache and hasattr(embedder, "cache") and total_chunks > 0:
            cached_map, missing_items = embedder.cache.get_many(texts, embedder.cache_model_name("document"))
            cache_hits = len(cached_map)
            cache_misses = len(missing_items)
            hit_pct = round(cache_hits / total_chunks * 100, 1)
            if cache_hits > 0:
                report(
                    58,
                    "A3",
                    f"Step A3: 임베딩 캐시 확인 ({cache_hits}개 재사용 / {cache_misses}개 신규 계산)",
                    f"캐시 적중률 {hit_pct}%",
                )

        try:
            if total_chunks == 0:
                report(85, "A3", "Step A3: 추가 임베딩 대상 청크 없음 (100% 중복 발췌본)", "임베딩 생략")
            elif total_chunks <= batch_size:
                embeddings = embedder.embed_documents(texts)
                report(85, "A3", f"Step A3: 임베딩 생성 완료 ({total_chunks}/{total_chunks} 청크)", "100%")
            else:
                for b_start in range(0, total_chunks, batch_size):
                    b_end = min(b_start + batch_size, total_chunks)
                    b_texts = texts[b_start:b_end]
                    b_embs = embedder.embed_documents(b_texts)
                    embeddings.extend(b_embs)

                    done_count = len(embeddings)
                    pct = 55 + int(30 * (done_count / total_chunks))
                    report(pct, "A3", f"Step A3: 임베딩 벡터 생성 중 ({done_count}/{total_chunks} 청크)...", f"{pct}%")
        except Exception as e:
            analysis = analyze_gemini_429_error(e)
            if analysis.get("is_429"):
                print(analysis["debug_message"])
                raise RuntimeError(
                    f"Gemini 임베딩 API 429 에러 발생: {analysis['title']}\n"
                    f"원인: {analysis['summary']}\n"
                    f"권장 조치: {analysis['recommended_action']}"
                ) from e
            raise
        embed_elapsed = round(time.time() - start_t, 3)

        dim = len(embeddings[0]) if embeddings else 0
        sample_vec = [round(float(v), 4) for v in embeddings[0][:6]] if embeddings else []
        norm = float(np.linalg.norm(embeddings[0])) if embeddings else 0.0

        # Step A3 문서 임베딩 비용 산출 및 과금 카운터 기록
        total_chars_to_embed = sum(len(t) for t in texts)
        cached_chars = sum(len(texts[idx]) for idx in cached_map.keys()) if (use_embedding_cache and hasattr(embedder, "cache") and total_chunks > 0 and 'cached_map' in locals()) else (total_chars_to_embed if cache_misses == 0 and cache_hits > 0 else 0)
        emb_job_res = self.cost_tracker.record_embedding_job(
            model_name=model_name,
            total_tokens_or_chars=total_chars_to_embed,
            cached_tokens_or_chars=cached_chars,
            job_type="index",
            doc_name=path.name,
        )

        cache_hit_ratio = round((cache_hits / max(1, total_chunks) * 100), 1) if use_embedding_cache else 0.0
        step_a3_data = {
            "step": "A3",
            "title": "임베딩하기 (청크 → 숫자 배열)",
            "model_name": model_name,
            "total_embedded": len(embeddings),
            "skipped_duplicates": total_chunks_cnt - len(chunks_to_embed),
            "dimension": dim,
            "elapsed_sec": embed_elapsed,
            "sample_vector_preview": sample_vec,
            "l2_norm": round(norm, 4),
            "cache_enabled": use_embedding_cache,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_ratio": cache_hit_ratio,
            "cached": (cache_misses == 0 and cache_hits > 0),
            "cost_estimation": emb_job_res,
            "token_usage": {
                "total_tokens": emb_job_res["total_tokens"],
                "cached_tokens": emb_job_res["cached_tokens"],
                "active_tokens": emb_job_res["active_tokens"],
            },
        }
        cache_suffix = f" (⚡ 캐시 {cache_hits}개 재사용 / {cache_hit_ratio}%)" if (use_embedding_cache and cache_hits > 0) else ""
        report(88, "A3", f"Step A3 완료: {len(embeddings)}개 청크 임베딩 완료 ({embed_elapsed}초){cache_suffix}", f"{dim}차원")

        # =====================================================================
        # Step A4. 벡터 스토어 저장 (벡터 + 원문 + 출처 연결)
        # =====================================================================
        a4_start_t = time.time()
        report(90, "A4", "Step A4: SQLite 벡터 스토어 저장 시작...", "테이블: langchain_documents")
        ids = [c.metadata["chunk_id"] for c in chunks_to_embed]
        self.vector_store.commit_pages(chunks_to_embed, ids, embeddings, model_name=model_name,
            source_id=source_id, page_hashes=page_hashes, reindex=reindex)

        db_stats = self.vector_store.get_stats()
        sample_rec = self.vector_store.get_sample_record()
        a4_elapsed = round(time.time() - a4_start_t, 3)
        total_index_elapsed = round(time.time() - pipeline_start_t, 3)

        step_a4_data = {
            "step": "A4",
            "title": "벡터 스토어 저장 (벡터 + 원문 + 출처 연결)",
            "db_path": db_stats["db_path"],
            "table_name": db_stats["table_name"],
            "total_records": db_stats["total_records"],
            "file_size_kb": round(db_stats["file_size_bytes"] / 1024, 1),
            "sample_record": sample_rec,
            "duplicate_diagnosis": dup_diagnosis,
            "duplicate_ratio": dup_ratio,
        }
        report(98, "A4", f"Step A4 완료: SQLite DB 저장 및 색인 완료 (총 {db_stats['total_records']}개)", f"{step_a4_data['file_size_kb']} KB")

        summary_msg = f"성공적으로 {len(chunks_to_embed)}개 청크를 SQLite에 저장했습니다."
        if dup_ratio >= 90.0:
            summary_msg += f" (⚠️ 기존 문서 발췌본 감지: 중복률 {dup_ratio}%)"
        elif duplicate_chunk_count > 0:
            summary_msg += f" (중복 청크 {duplicate_chunk_count}개 감지 / 중복률 {dup_ratio}%)"

        report(100, "COMPLETE", f"인덱싱 완료! {summary_msg}", "A1 → A4 완료")

        # ---------------------------------------------------------------------
        # 📝 RAG 최적 조합 분석용 실험 로그 기록 (Part A 인덱싱)
        # ---------------------------------------------------------------------
        exp_log = self.experiment_logger.log_indexing(
            file_name=path.name,
            file_path=str(path),
            file_size_kb=round(path.stat().st_size / 1024, 1),
            total_pages=total_pages,
            extracted_pages=len(documents),
            loader_type=loader_type,
            ocr_used=ocr_used,
            ocr_reason=ocr_reason or "",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            total_chunks=total_chunks_cnt,
            duplicate_chunk_count=duplicate_chunk_count,
            duplicate_ratio=dup_ratio,
            duplicate_diagnosis=dup_diagnosis,
            chunks_to_embed=len(chunks_to_embed),
            avg_chunk_length=round(sum(chunk_lens) / len(chunk_lens), 1) if chunk_lens else 0.0,
            embedding_model=model_name,
            embedding_dim=dim,
            cache_enabled=use_embedding_cache,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_ratio=cache_hit_ratio,
            total_elapsed_sec=total_index_elapsed,
            load_elapsed_sec=a1_elapsed,
            split_elapsed_sec=a2_elapsed,
            embed_elapsed_sec=embed_elapsed,
            store_elapsed_sec=a4_elapsed,
            total_tokens=emb_job_res["total_tokens"],
            cached_tokens=emb_job_res["cached_tokens"],
            active_tokens=emb_job_res["active_tokens"],
            cost_usd=emb_job_res["cost_usd"],
            cost_krw=emb_job_res["cost_krw"],
            db_total_records=db_stats["total_records"],
        )

        return {
            "status": "success",
            "message": summary_msg,
            "experiment_log": exp_log,
            "step_a1": step_a1_data,
            "step_a2": step_a2_data,
            "step_a3": step_a3_data,
            "step_a4": step_a4_data,
        }

    # -------------------------------------------------------------------------
    # PART B. 질문에 답하기 (런타임 질의응답)
    # -------------------------------------------------------------------------
    def run_query(self, *args, **kwargs):
        with self._operation_lock:
            return self._run_query(*args, **kwargs)

    def _run_query(
        self,
        question: str,
        top_k: int = 3,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_embedding_cache: bool = True,
    ) -> Dict[str, Any]:
        """B1~B4 단계를 순차 실행하고 각 단계별 중간 출력 데이터를 딕셔너리로 반환합니다."""
        query_pipeline_start_t = time.time()
        emb_name = embedding_model or self.default_embedding_model
        llm_name = llm_model or self.default_llm_model

        self.vector_store.assert_model(emb_name, require_documents=True)
        embedder = self.get_embedder(emb_name, use_cache=use_embedding_cache)
        llm = self.get_llm(llm_name)

        # =====================================================================
        # Step B1. 질문 입력 (질문도 임베딩)
        # =====================================================================
        start_t = time.time()
        was_cached = False
        if use_embedding_cache and hasattr(embedder, "cache"):
            was_cached = embedder.cache.get(question, embedder.cache_model_name("query")) is not None

        query_vector = embedder.embed_query(question)
        b1_elapsed = round(time.time() - start_t, 3)

        # Step B1 질문 임베딩 비용 산출 및 과금 카운터 기록
        q_chars = len(question)
        cached_chars = q_chars if was_cached else 0
        b1_emb_job = self.cost_tracker.record_embedding_job(
            model_name=emb_name,
            total_tokens_or_chars=q_chars,
            cached_tokens_or_chars=cached_chars,
            job_type="query",
            doc_name=question[:25],
        )

        sample_q_vec = [round(float(v), 4) for v in query_vector[:6]]
        step_b1_data = {
            "step": "B1",
            "title": "질문 입력 (질문도 임베딩)",
            "question": question,
            "dimension": len(query_vector),
            "elapsed_sec": b1_elapsed,
            "sample_vector": sample_q_vec,
            "cache_enabled": use_embedding_cache,
            "cached": was_cached,
            "cost_estimation": b1_emb_job,
            "token_usage": {
                "total_tokens": b1_emb_job["total_tokens"],
                "cached_tokens": b1_emb_job["cached_tokens"],
                "active_tokens": b1_emb_job["active_tokens"],
            },
        }

        # =====================================================================
        # Step B2. 관련 청크 검색 (하이브리드: Kiwi BM25 + Dense RRF 또는 Dense 단독)
        # =====================================================================
        start_t = time.time()
        # 기본적으로 고정밀 키워드/연도/고유명사 탐색을 위해 하이브리드 RRF 검색 사용
        scored_results = self.vector_store.hybrid_search_with_score(
            query=question, k=top_k, bm25_weight=1.5, dense_weight=1.0, rrf_k=20, query_vector=query_vector
        )
        b2_elapsed_ms = round((time.time() - start_t) * 1000, 2)

        retrieved_list = []
        for rank, (doc, score) in enumerate(scored_results, start=1):
            retrieved_list.append({
                "rank": rank,
                "score": round(score, 4),
                "score_pct": f"{score * 100:.1f}%",
                "page": doc.metadata.get("page", 1),
                "chunk_id": doc.metadata.get("id") or doc.metadata.get("chunk_id", f"c{rank}"),
                "source": doc.metadata.get("source", ""),
                "content": doc.page_content.strip(),
                "snippet": doc.page_content.strip()[:150] + "...",
            })

        step_b2_data = {
            "step": "B2",
            "title": "관련 청크 검색 (하이브리드 RRF: Kiwi BM25 + Dense Vector)",
            "top_k": top_k,
            "elapsed_ms": b2_elapsed_ms,
            "matches": retrieved_list,
        }

        # =====================================================================
        # Step B3. 프롬프트 구성 (질문 + 원문 + 지시문)
        # (원래 질문도 그대로 전달합니다)
        # =====================================================================
        context_parts = []
        for item in retrieved_list:
            part = (
                f"[참고 자료 #{item['rank']} | 출처: {item['page']}쪽 | ID: {item['chunk_id']}]\n"
                f"{item['content']}"
            )
            context_parts.append(part)
        context_text = "\n\n".join(context_parts)

        system_instruction = (
            "당신은 제공된 문서를 바탕으로 사용자에게 친절하고 신뢰성 높은 답변을 제공하는 AI 어시스턴트입니다.\n"
            "반드시 아래 [참고 자료]에 근거하여 사실만을 명확하게 답변하세요.\n"
            "문서에 언급되지 않은 정보는 지어내지 말고 솔직하게 모른다고 답하세요.\n"
            "답변 말미에는 답변의 근거가 된 출처 쪽수(예: [출처: X쪽])를 반드시 명시하세요."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{system_instruction}"),
            (
                "human",
                "다음 [참고 자료]를 읽고 질문에 답해주세요.\n\n"
                "[참고 자료]\n{context}\n\n"
                "[사용자 원래 질문]\n{question}",
            ),
        ])

        prompt_obj = prompt_template.invoke({
            "system_instruction": system_instruction,
            "context": context_text,
            "question": question,
        })

        rendered_prompt = "\n\n".join(
            [f"[{m.type.upper()}]\n{m.content}" for m in prompt_obj.messages]
        )

        step_b3_data = {
            "step": "B3",
            "title": "프롬프트 구성 (질문 + 원문 + 지시문)",
            "context_count": len(retrieved_list),
            "rendered_prompt": rendered_prompt,
            "prompt_length_chars": len(rendered_prompt),
            "original_question_retained": True,
        }

        # =====================================================================
        # Step B4. LLM -> 답변 (근거를 읽고 생성)
        # =====================================================================
        start_t = time.time()
        try:
            response = llm.invoke(prompt_obj)
        except Exception as e:
            analysis = analyze_gemini_429_error(e)
            if analysis.get("is_429"):
                print(analysis["debug_message"])
                raise RuntimeError(
                    f"API 429 에러 발생: {analysis['title']}\n"
                    f"원인: {analysis['summary']}\n"
                    f"권장 조치: {analysis['recommended_action']}"
                ) from e
            raise

        b4_elapsed = round(time.time() - start_t, 2)

        answer_text = extract_llm_text(response)
        citations = list(dict.fromkeys([f"{item['page']}쪽" for item in retrieved_list]))

        if any(k in llm_name.lower() for k in ("claude", "haiku")):
            provider_name = "Anthropic on Vertex AI"
        elif "deepseek" in llm_name.lower():
            provider_name = "Vertex AI Model Garden (DeepSeek)"
        elif ":" in llm_name or "ollama" in llm_name.lower():
            provider_name = "Ollama (로컬)"
        elif "gemma" in llm_name.lower():
            provider_name = "Vertex AI Model Garden (Gemma 4)"
        elif "lite" in llm_name.lower() or "gemini" in llm_name.lower():
            provider_name = "Vertex AI (ADC)"
        else:
            provider_name = "Vertex AI"

        # 토큰 카운팅 및 공식 요율 기반 과금 계산
        token_info = extract_tokens_from_response(
            response=response,
            prompt_text=rendered_prompt,
            answer_text=answer_text,
        )
        emb_cost_info = calculate_embedding_cost(
            model_name=emb_name,
            char_count_or_tokens=len(question),
            cached=was_cached,
        )
        tracking_result = self.cost_tracker.record_query(
            model_name=llm_name,
            question=question,
            input_tokens=token_info["input_tokens"],
            output_tokens=token_info["output_tokens"],
            cache_read_tokens=token_info["cache_read_tokens"],
            elapsed_sec=b4_elapsed,
            embedding_cost_usd=emb_cost_info["cost_usd"],
        )
        cost_estimation = tracking_result["query_cost"]
        cost_estimation["embedding_cost"] = emb_cost_info
        cost_estimation["combined_total_usd"] = tracking_result["combined_total_usd"]
        cost_estimation["combined_total_krw"] = tracking_result["combined_total_krw"]
        cost_estimation["formatted_combined_usd"] = f"${tracking_result['combined_total_usd']:.6f}"
        cost_estimation["formatted_combined_krw"] = f"약 {tracking_result['combined_total_krw']:.2f}원"

        step_b4_data = {
            "step": "B4",
            "title": "LLM → 답변 (근거를 읽고 생성)",
            "model_name": llm_name,
            "provider": provider_name,
            "answer": answer_text,
            "citations": citations,
            "elapsed_sec": b4_elapsed,
            "token_usage": token_info,
            "cost_estimation": cost_estimation,
        }

        total_query_elapsed = round(time.time() - query_pipeline_start_t, 3)

        # 현재 활성 문서명 및 DB 전체 청크 수 조회
        active_doc_name = ""
        if retrieved_list:
            active_doc_name = retrieved_list[0].get("file_name") or retrieved_list[0].get("source", "")
        if not active_doc_name:
            sample_rec = self.vector_store.get_sample_record()
            if sample_rec:
                active_doc_name = sample_rec.get("metadata", {}).get("file_name", "")

        db_stats = self.vector_store.get_stats()

        # ---------------------------------------------------------------------
        # 📝 RAG 최적 조합 분석용 실험 로그 기록 (Part B 질의응답)
        # ---------------------------------------------------------------------
        exp_log = self.experiment_logger.log_query(
            question=question,
            llm_model=llm_name,
            llm_provider=provider_name,
            embedding_model=emb_name,
            embedding_dim=len(query_vector),
            cache_enabled=use_embedding_cache,
            query_cache_hit=was_cached,
            top_k=top_k,
            retrieved_chunks_count=len(retrieved_list),
            citations=citations,
            answer_text=answer_text,
            total_elapsed_sec=total_query_elapsed,
            b1_embed_elapsed_sec=b1_elapsed,
            b2_retrieval_elapsed_ms=b2_elapsed_ms,
            b4_llm_elapsed_sec=b4_elapsed,
            input_tokens=token_info["input_tokens"],
            output_tokens=token_info["output_tokens"],
            cache_read_tokens=token_info["cache_read_tokens"],
            embedding_tokens=b1_emb_job["total_tokens"],
            llm_cost_usd=cost_estimation["total_cost_usd"],
            embedding_cost_usd=emb_cost_info["cost_usd"],
            total_cost_usd=tracking_result["combined_total_usd"],
            total_cost_krw=tracking_result["combined_total_krw"],
            active_doc_name=active_doc_name,
            total_indexed_chunks=db_stats.get("total_records", 0),
        )

        return {
            "status": "success",
            "answer": answer_text,
            "citations": citations,
            "token_usage": token_info,
            "cost_estimation": cost_estimation,
            "experiment_log": exp_log,
            "step_b1": step_b1_data,
            "step_b2": step_b2_data,
            "step_b3": step_b3_data,
            "step_b4": step_b4_data,
        }

