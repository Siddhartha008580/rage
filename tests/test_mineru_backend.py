import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.backends.mineru import MINERU_HEADER, convert_pages_with_mineru
from src.docling_fallback import find_pages_needing_docling, mineru_configuration


class MinerUBackendTests(unittest.TestCase):
    def test_mineru_is_enabled_by_default_and_can_be_disabled(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(mineru_configuration()["enabled"])
        with patch.dict(
            "os.environ", {"MINERU_ENABLED": "0"}, clear=True
        ):
            self.assertFalse(mineru_configuration()["enabled"])
        with patch.dict(
            "os.environ",
            {"MINERU_ENABLED": "true", "MINERU_TIER": "flash"},
            clear=True,
        ):
            config = mineru_configuration()
        self.assertTrue(config["enabled"])
        self.assertEqual(config["tier"], "flash")

    def test_enabling_mineru_changes_pdf_output_token(self) -> None:
        from src.ingest import output_token

        with patch.dict(
            "os.environ",
            {"MINERU_ENABLED": "0"},
            clear=False,
        ):
            disabled_token = output_token(Path("sample.pdf"), "a" * 64)
        with patch.dict(
            "os.environ",
            {"MINERU_ENABLED": "1"},
            clear=False,
        ):
            enabled_token = output_token(Path("sample.pdf"), "a" * 64)
        self.assertNotEqual(disabled_token, enabled_token)

    def test_mineru_can_enable_page_routing_without_docling(self) -> None:
        page = SimpleNamespace(
            rect=SimpleNamespace(
                x0=0, y0=0, x1=100, y1=100, width=100, height=100
            ),
            get_image_info=lambda: [{"bbox": (0, 0, 100, 100)}],
            get_text=lambda *args: (
                "x" * 20 if not args else []
            ),
        )
        with patch.dict(
            "os.environ",
            {"MINERU_ENABLED": "1", "DOCLING_ENABLED": "0"},
            clear=True,
        ):
            self.assertEqual(find_pages_needing_docling([page]), {0: (
                "full-page image with sparse text (100% coverage, 20 chars)"
            )})

    def test_selected_page_is_converted_with_mineru_cli(self) -> None:
        class Document:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __len__(self):
                return 1

        class SinglePage:
            def insert_pdf(self, *_args, **_kwargs):
                pass

            def save(self, path):
                Path(path).touch()

            def close(self):
                pass

        pymupdf = SimpleNamespace(
            open=lambda *_args: Document() if _args else SinglePage()
        )
        process = MagicMock()
        process.pid = 1
        process.returncode = 0
        process.wait.return_value = 0

        def launch(command, **_kwargs):
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text("# Extracted by MinerU", encoding="utf-8")
            return process

        with (
            patch.dict(sys.modules, {"pymupdf": pymupdf}),
            patch("src.backends.mineru.shutil.which", return_value="/usr/bin/mineru-kit"),
            patch("src.backends.mineru.subprocess.Popen", side_effect=launch) as popen,
        ):
            result = convert_pages_with_mineru(
                Path("sample.pdf"), {0}, [], tier="flash"
            )

        self.assertEqual(result, {0: "# Extracted by MinerU"})
        command = popen.call_args.args[0]
        self.assertEqual(command[1], "parse")
        self.assertEqual(command[-1], "flash")
        self.assertTrue(MINERU_HEADER.startswith("<!-- rendered-by: mineru -->"))


if __name__ == "__main__":
    unittest.main()
