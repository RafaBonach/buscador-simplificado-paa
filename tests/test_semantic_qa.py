import tempfile
import unittest
from pathlib import Path

from semantic_qa import CorpusManager, chunk_text


class TestChunking(unittest.TestCase):
    def test_chunk_text_with_overlap(self):
        text = "abcdefghij"
        chunks = chunk_text(text, chunk_size_chars=4, overlap_chars=1)
        self.assertEqual(chunks, ["abcd", "defg", "ghij"])

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text("   ", chunk_size_chars=5, overlap_chars=0), [])


class TestCorpusManager(unittest.TestCase):
    def test_load_chunks_from_txt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "doc.txt"
            file_path.write_text("texto de teste " * 20, encoding="utf-8")

            chunks = CorpusManager.load_chunks_from_txt(Path(tmp_dir), chunk_size_chars=40, overlap_chars=10)
            self.assertGreater(len(chunks), 1)
            self.assertTrue(chunks[0].chunk_id.endswith("::0"))
            self.assertEqual(chunks[0].source, file_path.as_posix())


if __name__ == "__main__":
    unittest.main()
