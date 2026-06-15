"""
Tests for scripts/data/scraper/adapters/supp_fetcher.py

All HTTP calls are mocked — no network required.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import io

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data.scraper.adapters.supp_fetcher import (
    _resolve_url, _file_ext, fetch_supplementary_texts,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session(response_bytes: bytes = b"", status: int = 200,
                  content_length: int = 0):
    """Return a mock requests.Session whose GET returns fixed bytes."""
    session = MagicMock(spec=requests.Session)

    head_resp = MagicMock()
    head_resp.headers = {"Content-Length": str(content_length)}
    head_resp.raise_for_status = MagicMock()
    session.head.return_value = head_resp

    get_resp = MagicMock()
    get_resp.status_code = status
    get_resp.raise_for_status = MagicMock(
        side_effect=(requests.HTTPError() if status >= 400 else None)
    )
    # iter_content yields the bytes in one chunk
    get_resp.iter_content = MagicMock(return_value=iter([response_bytes]))
    session.get.return_value = get_resp

    return session


def _make_limiter():
    m = MagicMock()
    m.wait = MagicMock()
    return m


# ── _resolve_url ──────────────────────────────────────────────────────────────

class TestResolveUrl(unittest.TestCase):

    def test_absolute_url_returned_unchanged(self):
        url = "https://example.com/supp/file.xlsx"
        self.assertEqual(_resolve_url(url, "https://base.example/"), url)

    def test_bare_filename_joined_to_base(self):
        result = _resolve_url("table1.xlsx", "https://ncbi.nlm.nih.gov/pmc/articles/PMC123/bin/")
        self.assertEqual(result, "https://ncbi.nlm.nih.gov/pmc/articles/PMC123/bin/table1.xlsx")

    def test_relative_path_joined_to_base(self):
        result = _resolve_url("supp/data.csv", "https://ncbi.nlm.nih.gov/bin/")
        self.assertIn("data.csv", result)

    def test_http_absolute_returned_unchanged(self):
        url = "http://example.com/file.csv"
        self.assertEqual(_resolve_url(url, "https://base/"), url)


# ── _file_ext ─────────────────────────────────────────────────────────────────

class TestFileExt(unittest.TestCase):

    def test_xlsx(self):
        self.assertEqual(_file_ext("https://example.com/file.xlsx"), ".xlsx")

    def test_csv_with_query(self):
        self.assertEqual(_file_ext("https://example.com/data.csv?download=1"), ".csv")

    def test_pdf(self):
        self.assertEqual(_file_ext("https://example.com/paper.PDF"), ".pdf")

    def test_no_extension(self):
        self.assertEqual(_file_ext("https://example.com/noext"), "")

    def test_uppercase_normalised(self):
        self.assertEqual(_file_ext("https://example.com/FILE.XLSX"), ".xlsx")


# ── fetch_supplementary_texts: filtering ─────────────────────────────────────

class TestFetchSupplementaryTexts(unittest.TestCase):

    def test_empty_supp_urls_returns_empty(self):
        result = fetch_supplementary_texts(
            supp_urls=[], base_url="https://base/",
            session=_make_session(), limiter=_make_limiter(),
        )
        self.assertEqual(result, [])

    def test_unsupported_extension_skipped(self):
        session = _make_session(b"data", content_length=4)
        result = fetch_supplementary_texts(
            supp_urls=["https://example.com/image.png"],
            base_url="https://base/",
            session=session, limiter=_make_limiter(),
        )
        self.assertEqual(result, [])
        session.get.assert_not_called()

    def test_zip_skipped(self):
        session = _make_session(b"data", content_length=4)
        result = fetch_supplementary_texts(
            supp_urls=["https://example.com/supp.zip"],
            base_url="https://base/",
            session=session, limiter=_make_limiter(),
        )
        self.assertEqual(result, [])
        session.get.assert_not_called()

    def test_oversized_file_skipped(self):
        """Files larger than max_file_bytes should not be downloaded."""
        session = _make_session(b"x" * 100, content_length=100)
        result = fetch_supplementary_texts(
            supp_urls=["https://example.com/huge.xlsx"],
            base_url="https://base/",
            session=session, limiter=_make_limiter(),
            max_file_bytes=10,   # very small limit
        )
        self.assertEqual(result, [])
        session.get.assert_not_called()

    def test_http_error_skipped_gracefully(self):
        session = _make_session(b"", status=404)
        result = fetch_supplementary_texts(
            supp_urls=["https://example.com/missing.xlsx"],
            base_url="https://base/",
            session=session, limiter=_make_limiter(),
        )
        self.assertEqual(result, [])

    def test_max_files_limit_respected(self):
        """No more than max_files files should be processed."""
        # Build a real xlsx bytes using openpyxl if available, else skip
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl not installed")

        import io as _io
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["sequence", "target"])
        ws.append(["ATCGATCGATCGATCGATCG", "Thrombin"])
        buf = _io.BytesIO()
        wb.save(buf)
        xlsx_bytes = buf.getvalue()

        session = _make_session(xlsx_bytes, content_length=len(xlsx_bytes))

        urls = [f"https://example.com/supp{i}.xlsx" for i in range(10)]
        result = fetch_supplementary_texts(
            supp_urls=urls,
            base_url="https://base/",
            session=session, limiter=_make_limiter(),
            max_files=3,
        )
        # max_files=3 means at most 3 text blocks returned
        self.assertLessEqual(len(result), 3)

    def test_relative_href_resolved_before_fetch(self):
        """Relative hrefs must be joined to base_url before GET."""
        session = _make_session(b"ATCGATCGATCGATCGATCG", content_length=20)
        # Patch _parse_to_text so we don't need real parsers
        with patch("scripts.data.scraper.adapters.supp_fetcher._parse_to_text",
                   return_value="ATCGATCGATCGATCGATCG"):
            fetch_supplementary_texts(
                supp_urls=["table1.csv"],
                base_url="https://ncbi.nlm.nih.gov/pmc/articles/PMC999/bin/",
                session=session, limiter=_make_limiter(),
            )
        # GET must have been called with the resolved absolute URL
        called_url = session.get.call_args[0][0]
        self.assertIn("PMC999", called_url)
        self.assertIn("table1.csv", called_url)

    def test_rate_limiter_called_for_each_file(self):
        """limiter.wait() must be called for every download attempt."""
        limiter = _make_limiter()
        with patch("scripts.data.scraper.adapters.supp_fetcher._parse_to_text",
                   return_value="text"):
            session = _make_session(b"data", content_length=4)
            fetch_supplementary_texts(
                supp_urls=["https://example.com/a.csv", "https://example.com/b.csv"],
                base_url="https://base/",
                session=session, limiter=limiter,
            )
        # wait() called at least twice (HEAD + GET per file)
        self.assertGreaterEqual(limiter.wait.call_count, 2)

    def test_parser_exception_does_not_crash_fetcher(self):
        """If _parse_to_text raises, the file is skipped and iteration continues."""
        session = _make_session(b"corrupted", content_length=9)
        with patch("scripts.data.scraper.adapters.supp_fetcher._parse_to_text",
                   side_effect=Exception("corrupt file")):
            result = fetch_supplementary_texts(
                supp_urls=["https://example.com/bad.xlsx",
                           "https://example.com/good.xlsx"],
                base_url="https://base/",
                session=session, limiter=_make_limiter(),
            )
        # Both files attempted; both yielded None/exception → empty result
        self.assertEqual(result, [])

    def test_csv_content_returned_as_text(self):
        """A valid CSV should be parsed and its text returned."""
        csv_bytes = b"sequence,target\nATCGATCGATCGATCGATCG,Thrombin\n"
        session = _make_session(csv_bytes, content_length=len(csv_bytes))
        result = fetch_supplementary_texts(
            supp_urls=["https://example.com/data.csv"],
            base_url="https://base/",
            session=session, limiter=_make_limiter(),
        )
        self.assertEqual(len(result), 1)
        self.assertIn("ATCGATCGATCGATCGATCG", result[0])

    def test_excel_file_parsed(self):
        """A real in-memory XLSX should yield text containing the sequence."""
        try:
            import openpyxl
        except ImportError:
            self.skipTest("openpyxl not installed")

        import io as _io
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["sequence", "target"])
        ws.append(["ATCGATCGATCGATCGATCG", "Thrombin"])
        buf = _io.BytesIO()
        wb.save(buf)
        xlsx_bytes = buf.getvalue()

        session = _make_session(xlsx_bytes, content_length=len(xlsx_bytes))
        result = fetch_supplementary_texts(
            supp_urls=["https://example.com/supp.xlsx"],
            base_url="https://base/",
            session=session, limiter=_make_limiter(),
        )
        self.assertEqual(len(result), 1)
        self.assertIn("ATCGATCGATCGATCGATCG", result[0])


# ── pubmed_pmc integration: supp_fetcher is called ───────────────────────────

class TestPubMedAdapterIntegration(unittest.TestCase):
    """
    Verify that PubMedPMCAdapter calls fetch_supplementary_texts when
    the parsed NXML has supplementary_urls.
    """

    def test_supp_fetcher_called_when_supp_urls_present(self):
        from scripts.data.scraper.adapters.pubmed_pmc import PubMedPMCAdapter
        from scripts.data.scraper.parsers.xml_parser import ParsedXML

        adapter = PubMedPMCAdapter()

        fake_doc = ParsedXML(
            title="Test paper",
            abstract="DNA aptamer for thrombin",
            body_text="",
            full_text="DNA aptamer for thrombin",
            supplementary_urls=["supp_table1.xlsx"],
            parse_ok=True,
        )

        with patch.object(adapter, "_entrez_search", return_value=["12345678"]), \
             patch.object(adapter, "_entrez_fetch_abstracts", return_value={"12345678": ""}), \
             patch.object(adapter, "_pmids_to_pmcids", return_value={"12345678": "PMC9999999"}), \
             patch.object(adapter, "_entrez_fetch_pmc_xml", return_value=b"<xml/>"), \
             patch("scripts.data.scraper.adapters.pubmed_pmc.parse_nxml",
                   return_value=fake_doc), \
             patch("scripts.data.scraper.adapters.pubmed_pmc.fetch_supplementary_texts",
                   return_value=[]) as mock_fetch:
            adapter.run(max_results=10)

        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args
        self.assertIn("supp_table1.xlsx", call_kwargs.kwargs["supp_urls"])
        self.assertIn("PMC9999999", call_kwargs.kwargs["base_url"])

    def test_supp_fetcher_not_called_when_no_supp_urls(self):
        """If supplementary_urls is empty, no fetch should happen."""
        from scripts.data.scraper.adapters.pubmed_pmc import PubMedPMCAdapter
        from scripts.data.scraper.parsers.xml_parser import ParsedXML

        adapter = PubMedPMCAdapter()
        fake_doc = ParsedXML(
            full_text="some text",
            supplementary_urls=[],   # empty
            parse_ok=True,
        )

        with patch.object(adapter, "_entrez_search", return_value=["12345678"]), \
             patch.object(adapter, "_entrez_fetch_abstracts", return_value={"12345678": ""}), \
             patch.object(adapter, "_pmids_to_pmcids", return_value={"12345678": "PMC9999999"}), \
             patch.object(adapter, "_entrez_fetch_pmc_xml", return_value=b"<xml/>"), \
             patch("scripts.data.scraper.adapters.pubmed_pmc.parse_nxml",
                   return_value=fake_doc), \
             patch("scripts.data.scraper.adapters.pubmed_pmc.fetch_supplementary_texts",
                   return_value=[]) as mock_fetch:
            adapter.run(max_results=10)

        mock_fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
