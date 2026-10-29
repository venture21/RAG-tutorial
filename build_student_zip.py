"""Build a data-free student ZIP from an explicit allowlist (standard library only)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = [
    "README.md", "RELEASE_NOTES.txt", "pyproject.toml", "uv.lock", "requirements.txt",
    ".env.example", ".gitignore", "app_rag.py", "rag_service.py", "runtime_config.py",
    "rag_step_by_step.py", "sqlite_vector_store.py", "embedding_cache.py",
    "gemini_pdf_loader.py", "gemini_quota_debugger.py", "cost_calculator.py",
    "rag_experiment_logger.py", "check_setup.py", "run_offline_checks.py", "offline_network.py",
    "verify_adc_auth.py", "test_embedding_cache.py", "test_student_release.py",
    "build_student_zip.py", "static/rag_dashboard.html", "pdf/README.txt",
    "docs/INSTALL_MACOS.txt", "docs/INSTALL_LINUX.txt", "docs/INSTALL_WINDOWS.txt",
    "ref_code/README.txt", "ref_code/claude_haiku_chat.py", "ref_code/deepseek_gcp_chat.py",
    "ref_code/gemma4_gcp_chat.py", "ref_code/gemini_flash_lite_chat.py",
    "docs/GITHUB.txt", ".github/workflows/offline-checks.yml",
]
FORBIDDEN_PARTS = {".venv", ".cache", "__pycache__", ".ipynb_checkpoints", "data", "logs", "uploads", ".git", ".DS_Store"}
PATTERNS = {
    "google-key": rb"AIza[0-9A-Za-z_-]{30,}",
    "provider-key": rb"sk-(?:or-v1-|proj-)[A-Za-z0-9_-]{20,}",
    "private-key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "oauth-token": rb"ya29\.[A-Za-z0-9._-]{20,}",
    "credential-url": rb"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s\"<>]{1,200}:[^\s\"<>@]{1,200}@",
    "personal-home": rb"/Users/[A-Za-z0-9_-]+/|[A-Za-z]:\\Users\\[^\\\s]+\\",
}


def inspect_entry(name, data):
    parts = PurePosixPath(name).parts
    if any(p in FORBIDDEN_PARTS for p in parts) or ".." in parts:
        raise ValueError(f"Excluded runtime path: {name}")
    filename = parts[-1]
    if filename.startswith(".env") and filename != ".env.example":
        raise ValueError(f"Actual environment file: {name}")
    if filename.lower().endswith((".pdf", ".db", ".sqlite", ".sqlite3", ".bin", ".pem", ".key")) or ".db-" in filename.lower() or data.startswith(b"%PDF-"):
        raise ValueError(f"PDF/database/binary credential file: {name}")
    for label, pattern in PATTERNS.items():
        if re.search(pattern, data):
            # Never echo the matched value.
            raise ValueError(f"Sensitive pattern {label} in {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/rag-student-studio-1.2.0.zip")
    args = parser.parse_args()
    payload = {}
    for name in FILES:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing file or symlink: {name}")
        content = path.read_bytes()
        inspect_entry(name, content)
        payload[name] = content
    manifest = {"release":"1.2.0", "date":"2026-10-29", "root":"rag-student-studio", "pdf_count":0,
        "excluded":"Actual environment files, credentials, PDFs, databases, caches, logs, uploads, virtual environments, notebooks, screenshots and historical evaluation data",
        "files":[{"path":name,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()} for name,content in sorted(payload.items())]}
    payload["MANIFEST.json"] = json.dumps(manifest,ensure_ascii=False,indent=2).encode("utf-8")
    output = args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name,content in sorted(payload.items()):
                entry = zipfile.ZipInfo("rag-student-studio/" + name, date_time=(2026,10,29,0,0,0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry,content)
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None
            for entry in archive.infolist():
                inspect_entry(entry.filename,archive.read(entry))
            for item in manifest["files"]:
                assert hashlib.sha256(archive.read("rag-student-studio/" + item["path"])).hexdigest() == item["sha256"]
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n",encoding="ascii")
    print(json.dumps({"zip":str(output),"files":len(payload),"bytes":output.stat().st_size,"pdf_count":0,"sha256":checksum},ensure_ascii=False))


if __name__ == "__main__":
    main()
