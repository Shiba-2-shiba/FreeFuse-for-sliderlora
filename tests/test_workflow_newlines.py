"""Exercise generator bytes with Windows-style default text translation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hybrid_newlines", ROOT / "tools/build_hybrid_validation_workflows.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class WorkflowNewlineTests(unittest.TestCase):
    def test_generated_json_stays_lf_with_windows_default_translation(self):
        original_factory = tempfile.NamedTemporaryFile

        def windows_default_factory(*args, **kwargs):
            # Real files, with CRLF translation only if production leaves it implicit.
            kwargs.setdefault("newline", "\r\n")
            return original_factory(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(generator.tempfile, "NamedTemporaryFile", side_effect=windows_default_factory):
                outputs = generator.write_workflows(output_dir=folder)
            self.assertEqual(len(outputs), 5)
            for output in outputs:
                with self.subTest(path=output):
                    data = Path(output).read_bytes()
                    self.assertNotIn(b"\r\n", data)
                    self.assertTrue(data.endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
