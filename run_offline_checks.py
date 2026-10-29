"""Run regression checks without ADC, developer .env files, or network access."""
import argparse
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from offline_network import offline_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with patch.dict(os.environ, {"LANGSMITH_TRACING":"false", "LANGCHAIN_TRACING_V2":"false"}), \
         patch("dotenv.load_dotenv", return_value=False), \
         offline_network():
        suite = unittest.defaultTestLoader.loadTestsFromNames(["verify_adc_auth", "test_embedding_cache", "test_student_release"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"network":"blocked","success":result.wasSuccessful()},indent=2),encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
