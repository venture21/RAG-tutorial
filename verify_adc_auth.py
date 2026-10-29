"""Verify ADC-only RAG authentication without credentials or external requests.

Run from the project root:
    .venv/bin/python RAG-tutorial/verify_adc_auth.py
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError
from offline_network import offline_network

# Importing the service must not read a developer's .env file during this check.
with patch("dotenv.load_dotenv", return_value=False):
    import gemini_pdf_loader
    import rag_service


class OfflineCredentials(Credentials):
    def refresh(self, request):
        raise AssertionError("Credential refresh is forbidden in offline checks")


class ADCOnlyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name)
        self.credentials = OfflineCredentials()
        self.environment = {
            "GOOGLE_API_KEY": "offline-google-key",
            "GEMINI_API_KEY": "offline-gemini-key",
            "GOOGLE_CLOUD_PROJECT": "offline-project",
            "GOOGLE_CLOUD_LOCATION": "global",
            # Explicit Vertex selection must override this ambient setting.
            "GOOGLE_GENAI_USE_VERTEXAI": "false",
        }
        self.enterContext(patch.dict(os.environ, self.environment, clear=True))
        self.default = self.enterContext(
            patch("google.auth.default", return_value=(self.credentials, "detected-project"))
        )
        self.enterContext(offline_network())

    def service_without_database(self):
        service = object.__new__(rag_service.RAGService)
        service.default_embedding_model = "text-embedding-004"
        service.default_llm_model = "gemini-3.8-flash"
        service._embedders = {}
        service._llms = {}
        service.active_auth_method = None
        service.data_dir = self.data_dir
        return service

    def assert_adc_client(self, client, project="offline-project"):
        self.addCleanup(client.close)
        sdk = client._api_client
        self.assertTrue(sdk.vertexai)
        self.assertIsNone(sdk.api_key)
        self.assertIs(sdk._credentials, self.credentials)
        self.assertEqual(sdk.project, project)
        self.assertIn("aiplatform.googleapis.com", sdk._http_options.base_url)
        for key, value in self.environment.items():
            if key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                self.assertEqual(os.environ[key], value, "Ambient API keys must remain unchanged")

    def test_service_startup_and_embedder_cache_use_adc(self):
        with tempfile.TemporaryDirectory(prefix="rag-adc-check-") as directory:
            with patch.object(rag_service.GoogleGenerativeAIEmbeddings, "embed_query", return_value=[0.1, 0.2]) as ping:
                service = rag_service.RAGService(db_path=str(Path(directory) / "rag.db"))
                embedder = service.get_embedder()
                self.assertIs(service.get_embedder(), embedder)
                ping.assert_not_called()
                self.assert_adc_client(embedder.client)
                self.assertIn("Vertex AI ADC", service.active_auth_method)

    def test_gemini_and_ocr_ignore_both_ambient_keys(self):
        service = self.service_without_database()
        llm = service.get_llm()
        self.assertIs(service.get_llm(), llm)
        self.assert_adc_client(llm.client)
        self.assert_adc_client(gemini_pdf_loader.get_genai_client())

    def test_adc_discovery_failure_never_falls_back_to_api_key(self):
        failure = DefaultCredentialsError("offline ADC discovery failure")
        with patch("google.auth.default", side_effect=failure):
            with patch.object(rag_service, "GoogleGenerativeAIEmbeddings") as embeddings, patch.object(
                rag_service, "ChatGoogleGenerativeAI"
            ) as llm, patch("google.genai.Client") as ocr:
                service = self.service_without_database()
                for operation in (service.get_embedder, service.get_llm, gemini_pdf_loader.get_genai_client):
                    with self.subTest(operation=operation.__name__):
                        with self.assertRaises(DefaultCredentialsError) as result:
                            operation()
                        self.assertIs(result.exception, failure)
                embeddings.assert_not_called()
                llm.assert_not_called()
                ocr.assert_not_called()
                self.assertFalse(service._embedders)
                self.assertFalse(service._llms)

    def test_client_initialization_failure_is_not_retried_with_api_key(self):
        cases = (
            ("embedding", "rag_service.GoogleGenerativeAIEmbeddings"),
            ("llm", "rag_service.ChatGoogleGenerativeAI"),
            ("ocr", "google.genai.Client"),
        )
        for component, target in cases:
            with self.subTest(component=component):
                service = self.service_without_database()
                operation = {
                    "embedding": service.get_embedder,
                    "llm": service.get_llm,
                    "ocr": gemini_pdf_loader.get_genai_client,
                }[component]
                failure = RuntimeError("offline client initialization failure")
                with patch(target, side_effect=failure) as constructor:
                    with self.assertRaises(RuntimeError) as result:
                        operation()
                    self.assertIs(result.exception, failure)
                    constructor.assert_called_once()
                self.assertFalse(service._embedders)
                self.assertFalse(service._llms)

    def test_initialization_does_not_make_embedding_request(self):
        service = self.service_without_database()
        failure = RuntimeError("offline model or region failure")
        embedder = Mock()
        embedder.embed_query.side_effect = failure
        with patch.object(rag_service, "GoogleGenerativeAIEmbeddings", return_value=embedder) as constructor:
            service.get_embedder()
            constructor.assert_called_once()
            embedder.embed_query.assert_not_called()

    def test_missing_project_fails_without_using_hardcoded_project(self):
        os.environ.pop("GOOGLE_CLOUD_PROJECT")
        with patch("google.auth.default", return_value=(self.credentials, None)):
            service = self.service_without_database()
            for operation in (service.get_embedder, service.get_llm, gemini_pdf_loader.get_genai_client):
                with self.subTest(operation=operation.__name__):
                    with self.assertRaisesRegex(ValueError, "GOOGLE_CLOUD_PROJECT"):
                        operation()

    def test_project_detected_by_adc_is_supported(self):
        os.environ.pop("GOOGLE_CLOUD_PROJECT")
        service = self.service_without_database()
        with patch.object(rag_service.GoogleGenerativeAIEmbeddings, "embed_query", return_value=[0.1]):
            self.assert_adc_client(service.get_embedder().client, "detected-project")
        self.assert_adc_client(service.get_llm().client, "detected-project")
        self.assert_adc_client(gemini_pdf_loader.get_genai_client(), "detected-project")

    def test_ollama_selection_stays_local_without_adc(self):
        service = self.service_without_database()
        local = Mock()
        with patch("google.auth.default", side_effect=DefaultCredentialsError("ADC unavailable")) as default:
            with patch.object(rag_service, "ChatOllama", return_value=local) as constructor:
                self.assertIs(service.get_llm("gemma4:e2b"), local)
                self.assertIs(service.get_llm("gemma4:e2b"), local)
                constructor.assert_called_once_with(model="gemma4:e2b", temperature=0.2)
                default.assert_not_called()

    def test_anthropic_vertex_client_constructs_offline(self):
        service = self.service_without_database()
        client = service.get_llm("claude-haiku-5-5")
        self.assertIs(service.get_llm("claude-haiku-5-5"), client)


if __name__ == "__main__":
    unittest.main(verbosity=2)
