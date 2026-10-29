"""Offline regressions. Only synthetic documents and temporary data are used."""
import io
import json
import os
import socket
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from offline_network import offline_network

with patch("dotenv.load_dotenv", return_value=False):
    import rag_service
    from app_rag import create_app
    from embedding_cache import CachedEmbeddings, EmbeddingCache
    from gemini_pdf_loader import GeminiFlashPDFLoader
    import sqlite_vector_store
from fastapi.testclient import TestClient
from google.auth.credentials import AnonymousCredentials
import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


def make_pdf(path, texts):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        if text:
            font = DictionaryObject({NameObject("/Type"):NameObject("/Font"), NameObject("/Subtype"):NameObject("/Type1"), NameObject("/BaseFont"):NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):DictionaryObject({NameObject("/F1"):writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 20 700 Td (" + text.encode("ascii") + b") Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


class FakeEmbeddings(Embeddings):
    def __init__(self, dimension=2):
        self.dimension = dimension
        self.queries = 0
        self.documents = 0

    def embed_documents(self, texts):
        self.documents += len(texts)
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]

    def embed_query(self, text):
        self.queries += 1
        return [0.0, 1.0] + [0.0] * (self.dimension - 2)


class OfflineNetworkTests(unittest.TestCase):
    def test_socketpair_bootstrap_works_with_guard(self):
        with offline_network():
            left, right = socket.socketpair()
            with left, right:
                left.settimeout(2)
                right.settimeout(2)
                left.sendall(b"offline")
                self.assertEqual(right.recv(7), b"offline")

    def test_direct_loopback_and_external_connections_remain_blocked(self):
        with offline_network():
            for address in [("127.0.0.1", 9), ("192.0.2.1", 443)]:
                with self.subTest(address=address):
                    with self.assertRaisesRegex(AssertionError, "forbid network"):
                        socket.create_connection(address)
                    with socket.socket() as connection:
                        for connect in (connection.connect, connection.connect_ex):
                            with self.assertRaisesRegex(AssertionError, "forbid network"):
                                connect(address)


class StudentReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf_dir = self.root / "pdf"
        self.pdf_dir.mkdir()
        self.enterContext(patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT":"offline-project", "GOOGLE_CLOUD_LOCATION":"global", "LANGSMITH_TRACING":"false", "LANGCHAIN_TRACING_V2":"false"}, clear=True))
        self.adc = self.enterContext(patch("google.auth.default", side_effect=AssertionError("Unexpected ADC access")))
        self.enterContext(offline_network())
        self.service = rag_service.RAGService(data_dir=self.root / "data", allowed_pdf_dirs=[self.pdf_dir, self.root / "data/uploads"])
        self.emb = FakeEmbeddings()
        self.service.get_embedder = Mock(return_value=self.emb)
        self.service.get_llm = Mock(return_value=Mock(invoke=Mock(return_value=AIMessage(content="offline answer", usage_metadata={"input_tokens":10,"output_tokens":2,"total_tokens":12}))))
        self.enterContext(patch.object(sqlite_vector_store, "HAS_HYBRID", False))
        self.client = TestClient(create_app(self.service), base_url="http://127.0.0.1:8000")
        self.pdf = self.pdf_dir / "exercise.pdf"
        make_pdf(self.pdf, ["UNCHANGED PAGE " * 8, "OLD PAGE " * 10])

    def index(self, **kwargs):
        return self.service.run_indexing(str(self.pdf), ocr_mode="pypdf", use_embedding_cache=False, **kwargs)

    def contents(self):
        with self.service.vector_store._get_connection() as conn:
            return [r[0] for r in conn.execute("SELECT content FROM langchain_documents ORDER BY id")]

    def test_startup_config_needs_no_credentials_or_network(self):
        self.assertEqual(self.client.get("/api/config").status_code, 200)
        self.adc.assert_not_called()
        self.service.get_embedder.assert_not_called()

    def test_outside_and_symlink_pdf_rejected(self):
        outside = self.root / "outside.pdf"
        make_pdf(outside, ["OUTSIDE " * 10])
        response = self.client.post("/api/pdf/pages", json={"pdf_path":str(outside),"ocr_mode":"pypdf"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("OUTSIDE", response.text)
        link = self.pdf_dir / "linked.pdf"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            return  # Windows without symlink privileges: outside-path test still ran.
        with self.assertRaises(ValueError):
            self.service.resolve_pdf(str(link))

    def test_foreign_origin_and_host_rejected(self):
        self.assertEqual(self.client.get("/api/config", headers={"Origin":"https://foreign.example"}).status_code,403)
        self.assertEqual(self.client.get("/api/config", headers={"Host":"foreign.example"}).status_code,400)
        self.assertEqual(self.client.get("/api/config", headers={"Origin":"http://127.0.0.1:8000"}).status_code,200)

    def test_upload_validates_pdf_and_windows_path(self):
        response = self.client.post("/api/upload", files={"file":("bad.pdf",b"not a pdf","application/pdf")})
        self.assertEqual(response.status_code,400)
        response = self.client.post("/api/upload", files={"file":("C:\\folder\\lesson.pdf",self.pdf.read_bytes(),"application/pdf")})
        self.assertEqual(response.status_code,200)
        self.assertEqual(Path(response.json()["file_path"]).parent,self.service.upload_dir)
        self.assertFalse(list(self.service.upload_dir.glob("*.tmp")))

    def test_invalid_overlap_and_page_limit_rejected(self):
        for body in [{"chunk_size":100,"chunk_overlap":150},{"max_pages":0},{"ocr_mode":"invalid"}]:
            self.assertEqual(self.client.post("/api/index",json={"pdf_path":str(self.pdf),**body}).status_code,422)

    def test_incremental_update_replaces_stale_chunks_and_repeat_is_idempotent(self):
        self.index(reindex=True)
        make_pdf(self.pdf,["UNCHANGED PAGE " * 8,"UPDATED PAGE " * 10])
        self.index(reindex=False)
        after = self.contents()
        self.assertEqual(len(after),2)
        self.assertFalse(any("OLD PAGE" in t for t in after))
        self.index(reindex=False)
        self.assertEqual(self.contents(),after)

    def test_partial_upload_keeps_absent_pages(self):
        self.index(reindex=True)
        make_pdf(self.pdf,["NEW FIRST PAGE " * 8])
        self.index(reindex=False)
        self.assertEqual(len(self.contents()),2)
        self.assertTrue(any("OLD PAGE" in t for t in self.contents()))

    def test_embedding_failure_leaves_old_index(self):
        self.index(reindex=True)
        before=self.contents()
        self.emb.embed_documents=Mock(side_effect=RuntimeError("offline embedding failure"))
        with self.assertRaises(RuntimeError):
            self.index(reindex=True)
        self.assertEqual(self.contents(),before)

    def test_atomic_storage_validation_preserves_old_index(self):
        self.index(reindex=True)
        before=self.contents()
        with self.assertRaises(ValueError):
            self.service.vector_store.commit_pages([Document(page_content="new")],["x"],[[1.0], [2.0]],model_name="other",source_id="x",page_hashes={1:"x"},reindex=True)
        self.assertEqual(self.contents(),before)

    def test_model_change_rejected_before_embedding(self):
        self.index(reindex=True)
        self.service.get_embedder.reset_mock()
        for operation in [lambda:self.index(reindex=False,embedding_model="other"),lambda:self.service.run_query("question",embedding_model="other")]:
            with self.assertRaises(ValueError):
                operation()
        self.service.get_embedder.assert_not_called()

    def test_insert_failure_rolls_back_page_deletion(self):
        import sqlite3
        self.index(reindex=True)
        before = self.contents()
        store = self.service.vector_store
        with store._get_connection() as conn:
            source = conn.execute("SELECT json_extract(metadata,'$.source_id') FROM langchain_documents LIMIT 1").fetchone()[0]
            conn.execute("CREATE TRIGGER abort_test BEFORE INSERT ON langchain_documents WHEN NEW.id='blocked' BEGIN SELECT RAISE(ABORT,'offline failure'); END")
            conn.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            store.commit_pages([Document(page_content="new",metadata={"page":1,"source_id":source})],["blocked"],[[1.0,0.0]],model_name="text-embedding-004",source_id=source,page_hashes={1:"new"})
        self.assertEqual(self.contents(),before)

    def test_query_vector_computed_once(self):
        self.index(reindex=True)
        result=self.service.run_query("question",use_embedding_cache=False)
        self.assertEqual(self.emb.queries,1)
        self.assertEqual(result["answer"],"offline answer")

    def test_document_and_query_cache_do_not_collide(self):
        cache=EmbeddingCache(self.root / "isolated-cache")
        wrapped=CachedEmbeddings(self.emb,cache=cache,model_name="fake")
        wrapped.embed_documents(["same text"])
        query=wrapped.embed_query("same text")
        self.assertEqual(query,[0.0,1.0])
        wrapped.embed_query("same text")
        self.assertEqual(self.emb.queries,1)

    def test_mixed_pdf_ocr_only_missing_page(self):
        make_pdf(self.pdf,["TEXT PAGE " * 12,""])
        ocr=Mock()
        ocr.load.return_value=[Document(page_content="SCANNED TEXT",metadata={"page":2})]
        with patch.object(rag_service,"GeminiFlashPDFLoader",return_value=ocr) as factory:
            result=self.service._extract_documents(self.pdf,ocr_mode="auto")
        self.assertEqual(factory.call_args.kwargs["page_indices"],[1])
        self.assertEqual([d.metadata["page"] for d in result[0]],[1,2])

    def test_ocr_failure_preserves_index(self):
        self.index(reindex=True)
        before=self.contents()
        make_pdf(self.pdf,[""])
        with patch.object(GeminiFlashPDFLoader,"_ocr_single_page",side_effect=RuntimeError("offline failure")):
            with self.assertRaisesRegex(RuntimeError,"OCR"):
                self.service.run_indexing(str(self.pdf),reindex=True,ocr_mode="auto",use_embedding_cache=False)
        self.assertEqual(self.contents(),before)

    def test_ocr_usage_counted_only_on_cache_miss(self):
        make_pdf(self.pdf,[""])
        loader=GeminiFlashPDFLoader(self.pdf,cache_dir=self.root / "ocr",usage_callback=self.service.cost_tracker.record_ocr_job)
        response=SimpleNamespace(text="OCR text",usage_metadata=SimpleNamespace(prompt_token_count=20,candidates_token_count=5,thoughts_token_count=2))
        with patch.object(loader,"get_client",return_value=Mock(models=Mock(generate_content=Mock(return_value=response)))):
            loader.load()
            loader.load()
        stats=self.service.cost_tracker.get_stats()
        self.assertEqual(stats["total_tokens"],27)
        self.assertEqual(stats["total_queries"],0)

    def test_model_garden_callable_refreshes_transport_authorization(self):
        from langchain_openai import ChatOpenAI
        seen=[]
        def respond(request):
            seen.append(request.headers["authorization"])
            return httpx.Response(200,json={"id":"offline","object":"chat.completion","created":0,"model":"offline","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}})
        client=httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        service=rag_service.RAGService(data_dir=self.root / "transport-data")
        with patch("google.auth.default",return_value=(AnonymousCredentials(),"offline-project")), patch.object(rag_service,"get_gcp_access_token",side_effect=["offline-a","offline-b"]), patch.object(rag_service,"ChatOpenAI",side_effect=lambda **kw:ChatOpenAI(**kw,http_client=client)):
            llm=service.get_llm("deepseek")
            llm.invoke("first")
            service.get_llm("deepseek").invoke("second")
        self.assertEqual(seen,["Bearer offline-a","Bearer offline-b"])

    def test_reference_models_use_callable_tokens_without_initial_refresh(self):
        import importlib
        for name, factory_name in [("deepseek_gcp_chat","create_deepseek_chat_model"),("gemma4_gcp_chat","create_gemma4_chat_model")]:
            with patch("dotenv.load_dotenv",return_value=False):
                module=importlib.import_module("ref_code."+name)
            with patch.object(module,"get_gcp_access_token",side_effect=AssertionError("Unexpected token refresh")) as token, patch.object(module,"ChatOpenAI") as constructor, patch("sys.stdout",new_callable=io.StringIO):
                getattr(module,factory_name)(project_id="offline-project")
            self.assertTrue(callable(constructor.call_args.kwargs["api_key"]))
            token.assert_not_called()

    def test_reference_env_path_is_program_root(self):
        import importlib.util
        root=Path(__file__).resolve().parent
        for path in (root/"ref_code").glob("*.py"):
            with patch("dotenv.load_dotenv",return_value=False) as dotenv:
                spec=importlib.util.spec_from_file_location("offline_"+path.stem,path)
                module=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            dotenv.assert_called_once_with(root/".env",override=False)


if __name__=="__main__":
    unittest.main(verbosity=2)
