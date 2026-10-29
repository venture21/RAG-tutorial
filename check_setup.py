"""Offline installation check; never prints keys or contacts Google."""
import importlib.util
import os
from pathlib import Path
import sqlite3
import sys
from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env", override=False)
    ok = (3,11) <= sys.version_info[:2] < (3,14)
    print("Python:", sys.version.split()[0], "OK" if ok else "3.11~3.13 필요 (3.11 권장)")
    modules = ["fastapi","uvicorn","pypdf","multipart","numpy","google.auth","google.genai","langchain_google_genai","langchain_google_vertexai","langchain_openai","langchain_ollama","kiwipiepy","rank_bm25","tzdata","anthropic"]
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            found = False
        print(name + ":", "OK" if found else "설치 필요")
        ok = ok and found
    with sqlite3.connect(":memory:") as connection:
        assert connection.execute("SELECT json_extract('{\"ok\":1}', '$.ok')").fetchone()[0] == 1
    print("SQLite JSON: OK")
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    print("프로젝트 설정:", "있음" if project and project != "YOUR_PROJECT_ID" else ".env에 본인 프로젝트 ID를 설정하세요")
    candidates = []
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        candidates.append(Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).expanduser())
    config = os.getenv("CLOUDSDK_CONFIG")
    if config:
        candidates.append(Path(config) / "application_default_credentials.json")
    if os.name == "nt" and os.getenv("APPDATA"):
        candidates.append(Path(os.environ["APPDATA"]) / "gcloud/application_default_credentials.json")
    else:
        candidates.append(Path.home() / ".config/gcloud/application_default_credentials.json")
    print("ADC 파일:", "있음 (유효성은 실제 요청 때 확인)" if any(p.is_file() for p in candidates) else "gcloud auth application-default login 필요")
    print("화면 확인은 ADC 없이 가능합니다. 실제 임베딩·OCR·클라우드 답변에는 본인 프로젝트와 ADC가 필요합니다.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
