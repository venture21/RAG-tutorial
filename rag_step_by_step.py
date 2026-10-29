"""PC 공통 CLI: 웹과 동일한 ADC 전용 RAG 파이프라인."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from rag_service import RAGService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="사용자가 준비한 PDF 경로")
    parser.add_argument("--auth", choices=["adc"], default="adc")
    parser.add_argument("--skip-indexing", action="store_true")
    parser.add_argument("--reindex", action="store_true", help="기존 전체 인덱스를 교체합니다")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--question", default="")
    parser.add_argument("--embedding-model", default="text-embedding-004")
    parser.add_argument("--llm-model", default="gemini-3.8-flash")
    parser.add_argument("--ocr-mode", choices=["auto","gemini_ocr","pypdf"], default="auto")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if not args.skip_indexing and not args.pdf:
        parser.error("--pdf 경로를 지정하세요. ZIP에는 PDF가 포함되지 않습니다.")
    if not 1 <= args.top_k <= 50:
        parser.error("top-k는 1~50이어야 합니다.")
    allowed = [args.pdf.resolve().parent] if args.pdf else None
    service = RAGService(allowed_pdf_dirs=allowed)
    if not args.skip_indexing:
        result = service.run_indexing(str(args.pdf.resolve()), chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap, embedding_model=args.embedding_model,
            reindex=args.reindex, ocr_mode=args.ocr_mode, max_pages=args.max_pages,
            use_embedding_cache=not args.no_cache)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    question = args.question
    while question or args.interactive:
        if not question:
            try:
                question = input("질문 (종료: /exit): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        if question == "/exit":
            break
        if question:
            result = service.run_query(question, top_k=args.top_k, llm_model=args.llm_model,
                embedding_model=args.embedding_model, use_embedding_cache=not args.no_cache)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        question = ""


if __name__ == "__main__":
    main()
