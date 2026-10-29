"""Gemini 3.8 Flash 기반 Multimodal PDF OCR Document Loader.

스캔본, 슬라이드, 이미지 기반 PDF에서 Gemini 3.8 Flash의 멀티모달 비전 기능을 활용하여
오직 '인쇄된 텍스트'만 정밀하게 전사(OCR)하고, 이미지 시각 묘사(사진 설명)는 배제하여
LangChain Document 리스트로 로드합니다.

특징:
1. 페이지별 병렬 처리 (ThreadPoolExecutor)로 빠른 로딩 속도
2. 디스크 캐시 (.cache/gemini_ocr/) 지원으로 동일 문서 재색인 시 0초 즉각 반환
3. LangChain BaseLoader 인터페이스 준수 (List[Document] 반환, page 메타데이터 유지)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader, PdfWriter

OCR_SYSTEM_INSTRUCTION = """당신은 한국어 및 다국어 문서의 정확한 OCR 전사 작업을 수행하는 전문 엔진입니다.
제공된 페이지를 읽고 인쇄되거나 필기된 텍스트만 원문 언어와 읽기 순서대로 정확하게 전사(OCR)하세요.

[필수 준수 규칙]
1. 텍스트만 추출: 본문, 장/절 제목, 소제목, 각주, 일반 표의 글자와 숫자, 캡션은 누락 없이 전사하세요.
2. 이미지 내용 제외: 사진, 그림, 삽화, 다이어그램 자체의 시각적 형태를 설명하거나 묘사하는 새 글(예: "이 그림은 ~를 나타낸다", "원형 그래프" 등)은 절대 작성하지 마세요. 오직 문서에 인쇄된 글자(텍스트)만 그대로 출력하세요.
3. 표와 서식 보존: 표는 행 단위로 작성하고 열은 공백 또는 탭으로 구분하여 수치와 텍스트를 빠뜨리지 마세요.
4. 원문 충실성: 내용을 요약, 번역, 윤문하거나 임의로 수정하지 마세요.
5. 부연 설명 금지: 마크다운 코드 블록(```)이나 인사말, 사과, 해설 없이 오직 추출된 본문 텍스트만 반환하세요.
6. 글자가 전혀 없는 순수 빈 페이지이거나 그림만 있고 글자가 없는 경우 빈 문자열을 반환하세요."""

OCR_PAGE_PROMPT_TEMPLATE = (
    "원본 PDF의 물리적 {page_num}쪽입니다. "
    "이 페이지에 인쇄된 글자(텍스트/제목/표/숫자)만 누락 없이 있는 그대로 OCR 추출하세요. "
    "사진이나 그림의 시각적 형태 묘사는 완전히 배제하세요."
)


def get_genai_client():
    """ADC 자격 증명만 사용하여 Vertex AI OCR 클라이언트를 반환합니다."""
    import google.auth
    from google import genai
    from google.genai import types

    options = types.HttpOptions(
        timeout=180000,
        retry_options=types.HttpRetryOptions(
            attempts=3,
            initial_delay=2,
            max_delay=20,
            http_status_codes=[429, 500, 502, 503, 504],
        ),
    )

    credentials, default_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or default_project
    if not project or project == "YOUR_PROJECT_ID":
        raise ValueError(
            "Google Cloud 프로젝트 ID를 찾을 수 없습니다. "
            "GOOGLE_CLOUD_PROJECT를 설정하고 ADC 로그인을 확인하세요."
        )
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    return genai.Client(
        vertexai=True,
        credentials=credentials,
        project=project,
        location=location,
        http_options=options,
    )


def compute_file_hash(path: Path) -> str:
    """PDF 파일의 내용 SHA-256 해시를 계산합니다."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class GeminiFlashPDFLoader(BaseLoader):
    """Gemini 3.8 Flash 비전 모델을 사용하여 PDF 페이지를 전사(OCR)하는 LangChain Loader."""

    def __init__(
        self,
        file_path: str | Path,
        model_name: str = "gemini-3.8-flash",
        max_workers: int = 4,
        max_pages: Optional[int] = None,
        page_start: int = 1,
        cache_dir: Optional[str | Path] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        system_instruction: Optional[str] = None,
        prompt_template: Optional[str] = None,
        page_indices: Optional[List[int]] = None,
        usage_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.file_path = Path(file_path).resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {self.file_path}")

        self.model_name = model_name
        self.max_workers = max(1, min(max_workers, 8))
        self.max_pages = max_pages
        self.page_start = max(1, page_start)
        self.progress_callback = progress_callback
        self.page_indices = page_indices
        self.usage_callback = usage_callback
        self.system_instruction = (system_instruction or "").strip() or OCR_SYSTEM_INSTRUCTION
        self.prompt_template = (prompt_template or "").strip() or OCR_PAGE_PROMPT_TEMPLATE

        base_cache = (
            Path(cache_dir)
            if cache_dir
            else (Path(__file__).resolve().parent / ".cache" / "gemini_ocr")
        )
        file_hash = compute_file_hash(self.file_path)
        # 프롬프트 내용이 달라지면 이전 캐시와 충돌하지 않도록 프롬프트 해시를 캐시 경로에 반영
        prompt_hash = hashlib.sha256(
            f"cache-v2\n{self.model_name}\n{self.system_instruction}\n{self.prompt_template}".encode("utf-8")
        ).hexdigest()[:8]
        self.cache_dir = base_cache / f"{self.file_path.stem}_{file_hash}_{prompt_hash}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()

    def get_client(self):
        if not hasattr(self._local, "client"):
            self._local.client = get_genai_client()
        return self._local.client

    def _get_page_bytes(self, reader: PdfReader, page_index: int) -> bytes:
        """pypdf를 사용하여 특정 페이지만을 포함하는 독립된 단일 페이지 PDF 바이트를 생성합니다."""
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def _ocr_single_page(
        self,
        reader: PdfReader,
        page_index: int,
        total_pages: int,
    ) -> Dict[str, Any]:
        """단일 페이지를 캐시에서 읽거나 Gemini 3.8 Flash로 OCR 수행합니다."""
        page_num = page_index + 1
        cache_file = self.cache_dir / f"page_{page_num:04d}.json"

        # 1. 캐시 확인
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if "text" in data:
                    if self.progress_callback:
                        self.progress_callback(page_num, total_pages, data["text"][:80])
                    return {
                        "page": page_num,
                        "text": data["text"],
                        "cached": True,
                        "elapsed_sec": data.get("elapsed_sec", 0.0),
                    }
            except Exception:
                pass

        # 2. Gemini 3.8 Flash OCR 호출
        from google.genai import types

        client = self.get_client()
        page_pdf_bytes = self._get_page_bytes(reader, page_index)
        prompt_text = (
            self.prompt_template.replace("{page_num}", str(page_num))
            if "{page_num}" in self.prompt_template
            else f"{self.prompt_template}\n(페이지: {page_num})"
        )

        t0 = time.time()
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=page_pdf_bytes, mime_type="application/pdf"),
                    prompt_text,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    max_output_tokens=8192,
                ),
            )
        except Exception as e:
            try:
                from gemini_quota_debugger import analyze_gemini_429_error
                analysis = analyze_gemini_429_error(e)
                if analysis.get("is_429"):
                    print(analysis["debug_message"])
            except Exception:
                pass
            raise

        elapsed = round(time.time() - t0, 2)
        extracted_text = (response.text or "").strip()
        usage = getattr(response, "usage_metadata", None)
        if self.usage_callback and usage:
            self.usage_callback(self.model_name, int(getattr(usage, "prompt_token_count", 0) or 0), int(getattr(usage, "candidates_token_count", 0) or 0) + int(getattr(usage, "thoughts_token_count", 0) or 0))

        # 3. 캐시 저장
        cache_data = {
            "page": page_num,
            "text": extracted_text,
            "char_count": len(extracted_text),
            "model": self.model_name,
            "elapsed_sec": elapsed,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_template": self.prompt_template,
        }
        temp_file = cache_file.with_suffix(".json.tmp")
        temp_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_file.replace(cache_file)

        if self.progress_callback:
            self.progress_callback(page_num, total_pages, extracted_text[:80])

        return {
            "page": page_num,
            "text": extracted_text,
            "cached": False,
            "elapsed_sec": elapsed,
        }

    def lazy_load(self) -> Iterator[Document]:
        """페이지별로 OCR을 수행하며 Document 객체를 순차 생성합니다."""
        reader = PdfReader(str(self.file_path))
        total_in_file = len(reader.pages)

        start_idx = self.page_start - 1
        if self.max_pages is not None:
            end_idx = min(start_idx + self.max_pages, total_in_file)
        else:
            end_idx = total_in_file

        target_indices = sorted(set(self.page_indices)) if self.page_indices is not None else list(range(start_idx, end_idx))
        if any(i < 0 or i >= total_in_file for i in target_indices):
            raise ValueError("OCR 페이지 범위가 올바르지 않습니다.")
        if not target_indices:
            return

        # 멀티스레딩 병렬 실행
        results: Dict[int, Dict[str, Any]] = {}
        failures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._ocr_single_page, reader, idx, total_in_file): idx
                for idx in target_indices
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results[idx] = res
                except Exception:
                    failures.append(idx + 1)

        if failures:
            raise RuntimeError(f"OCR 처리에 실패한 페이지: {sorted(failures)}. 기존 인덱스는 변경하지 않았습니다. 다시 시도하세요.")

        # 원본 페이지 순서대로 Document 방출
        for idx in target_indices:
            page_data = results.get(idx, {})
            text = page_data.get("text", "")
            page_num = idx + 1
            metadata = {
                "source": str(self.file_path.name),
                "path": str(self.file_path),
                "page": page_num,
                "total_pages": total_in_file,
                "ocr_model": self.model_name,
                "cached": page_data.get("cached", False),
            }
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """모든 페이지의 OCR 텍스트를 로드하여 List[Document]로 반환합니다."""
        return list(self.lazy_load())
