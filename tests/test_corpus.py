"""Tests for metadata preload and SPECTER2 corpus embedding."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from datagen.embed_corpus import (
    DIMENSION,
    build_faiss_index,
    create_document_encoder,
    embed_pending,
    parse_cuda_devices,
    normalize_database_url,
    pg_vector_literal,
    _contiguous_chunks,
)
from datagen.preload_metadata import (
    ArxivPaper,
    build_preload,
    canonical_arxiv_id,
    connect_database,
    is_computer_science,
    iter_snapshot,
    upload_preload,
    upsert_papers,
)
from search_candidates import ArxivIndex


class FakeEncoder:
    dimension = DIMENSION
    fingerprint = "fake-specter2-v1"

    @staticmethod
    def _vectors(texts):
        vectors = np.zeros((len(texts), DIMENSION), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, 0] = 0 if "attention" in text.lower() else 10
        return vectors

    def encode_documents(self, texts, batch_size):
        return self._vectors(texts)

    def encode_queries(self, texts, batch_size):
        return self._vectors(texts)


class PostgresUrlTests(unittest.TestCase):
    def test_normalize_adds_nebius_ca_for_verify_full(self):
        with tempfile.TemporaryDirectory() as directory:
            ca_path = Path(directory) / "root.crt"
            ca_path.write_text("CERT")
            with patch(
                "datagen.embed_corpus.ensure_nebius_msp_ca",
                return_value=ca_path,
            ):
                normalized = normalize_database_url(
                    "postgresql://u:p@host:5432/db?sslmode=verify-full"
                )
            self.assertIn("sslmode=verify-full", normalized)
            self.assertIn("sslrootcert=", normalized)
            self.assertNotIn("sslrootcert=system", normalized)

    def test_normalize_replaces_system_trust_with_ca_file(self):
        with tempfile.TemporaryDirectory() as directory:
            ca_path = Path(directory) / "root.crt"
            ca_path.write_text("CERT")
            with patch(
                "datagen.embed_corpus.ensure_nebius_msp_ca",
                return_value=ca_path,
            ):
                normalized = normalize_database_url(
                    "postgresql://u:p@host:5432/db?"
                    "sslmode=verify-full&sslrootcert=system"
                )
            self.assertIn("sslrootcert=", normalized)
            self.assertNotIn("sslrootcert=system", normalized)
            from urllib.parse import unquote

            self.assertIn(str(ca_path), unquote(normalized))


class PgVectorLiteralTests(unittest.TestCase):
    def test_pg_vector_literal_formats_floats(self):
        vector = np.array([0.1, 1.0, -2.5], dtype=np.float32)
        self.assertEqual(pg_vector_literal(vector), "[0.1,1,-2.5]")


class MultiGpuEncoderSelectionTests(unittest.TestCase):
    def test_parse_cuda_devices_accepts_all_empty_or_explicit_ids(self):
        self.assertEqual(parse_cuda_devices(None), [])
        self.assertEqual(parse_cuda_devices("all"), [])
        self.assertEqual(
            parse_cuda_devices("0,1,cuda:3"),
            ["cuda:0", "cuda:1", "cuda:3"],
        )

    def test_contiguous_chunks_preserve_order_for_uneven_batches(self):
        self.assertEqual(_contiguous_chunks(10, 3), [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(_contiguous_chunks(2, 8), [(0, 1), (1, 2)])

    @patch("datagen.embed_corpus.Specter2Encoder")
    @patch("datagen.embed_corpus.MultiGpuSpecter2Encoder")
    @patch("datagen.embed_corpus.available_cuda_devices")
    def test_create_document_encoder_uses_multi_gpu_when_available(
        self, available, multi_gpu, single_gpu
    ):
        available.return_value = ["cuda:0", "cuda:1"]
        encoder = create_document_encoder(device="cuda")
        self.assertIs(encoder, multi_gpu.return_value)
        multi_gpu.assert_called_once_with(["cuda:0", "cuda:1"], revision="main")
        single_gpu.assert_not_called()

    @patch("datagen.embed_corpus.Specter2Encoder")
    @patch("datagen.embed_corpus.MultiGpuSpecter2Encoder")
    @patch("datagen.embed_corpus.available_cuda_devices")
    def test_create_document_encoder_keeps_single_gpu_path(
        self, available, multi_gpu, single_gpu
    ):
        available.return_value = ["cuda:0"]
        encoder = create_document_encoder(device="cuda")
        self.assertIs(encoder, single_gpu.return_value)
        single_gpu.assert_called_once_with(device="cuda:0", revision="main")
        multi_gpu.assert_not_called()

    @patch("datagen.embed_corpus.Specter2Encoder")
    @patch("datagen.embed_corpus.MultiGpuSpecter2Encoder")
    @patch("datagen.embed_corpus.available_cuda_devices")
    def test_explicit_cuda_device_forces_single_gpu_path(
        self, available, multi_gpu, single_gpu
    ):
        available.return_value = ["cuda:0", "cuda:1"]
        encoder = create_document_encoder(device="cuda:0")
        self.assertIs(encoder, single_gpu.return_value)
        single_gpu.assert_called_once_with(device="cuda:0", revision="main")
        multi_gpu.assert_not_called()


class MetadataTests(unittest.TestCase):
    def test_canonicalizes_new_and_legacy_ids_without_corrupting_archive(self):
        self.assertEqual(canonical_arxiv_id("ARXIV:2001.01072v3"), "2001.01072")
        self.assertEqual(
            canonical_arxiv_id("oai:arXiv.org:solv-int/9701001v2"),
            "solv-int/9701001",
        )

    def test_cs_filter(self):
        self.assertTrue(is_computer_science("cs.LG stat.ML"))
        self.assertFalse(is_computer_science("hep-th math.AG"))

    def test_embedding_jobs_are_resumable_and_content_aware(self):
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            connection = connect_database(index_dir)
            papers = [
                ArxivPaper("1", "Attention Paper", "Channel attention."),
                ArxivPaper("2", "Geometry Paper", "Linear regions."),
            ]
            self.assertEqual(upsert_papers(connection, papers), 2)
            encoder = FakeEncoder()

            self.assertEqual(
                embed_pending(
                    connection,
                    index_dir,
                    encoder,
                    batch_size=2,
                    shard_size=2,
                ),
                2,
            )
            self.assertEqual(embed_pending(connection, index_dir, encoder), 0)
            upsert_papers(
                connection,
                [ArxivPaper("2", "Attention Geometry", "Updated abstract.")],
            )
            self.assertEqual(embed_pending(connection, index_dir, encoder), 1)
            connection.close()

    def test_snapshot_filters_cs_and_allows_missing_abstract(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "metadata.json"
            snapshot.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "1",
                                "title": "Title",
                                "categories": "cs.LG",
                            }
                        ),
                        json.dumps(
                            {
                                "id": "2",
                                "title": "Physics",
                                "abstract": "Abs",
                                "categories": "hep-th",
                            }
                        ),
                    ]
                )
                + "\n"
            )
            papers = list(iter_snapshot(snapshot, cs_only=True))
            self.assertEqual(len(papers), 1)
            self.assertEqual(papers[0].arxiv_id, "1")
            self.assertEqual(papers[0].abstract, "")

            snapshot.write_text(json.dumps({"title": "Missing ID"}) + "\n")
            with self.assertRaises(KeyError):
                next(iter_snapshot(snapshot, cs_only=False))

    def test_build_preload_writes_ready_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snap.jsonl"
            snapshot.write_text(
                json.dumps(
                    {
                        "id": "2001.00001",
                        "title": "A",
                        "abstract": "B",
                        "categories": "cs.AI",
                    }
                )
                + "\n"
            )
            local_dir = root / "preload"
            ready = build_preload(snapshot, local_dir, cs_only=True)
            self.assertEqual(ready["papers"], 1)
            self.assertTrue((local_dir / "metadata.sqlite").is_file())
            self.assertTrue((local_dir / "PRELOAD_READY.json").is_file())

    def test_upload_preload_puts_expected_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            local_dir = Path(directory)
            (local_dir / "metadata.sqlite").write_bytes(b"sqlite")
            (local_dir / "PRELOAD_READY.json").write_text("{}")
            client = MagicMock()
            with patch.dict(
                "os.environ",
                {
                    "NEBIUS_S3_BUCKET": "embedxiv-bucket",
                    "NEBIUS_S3_ENDPOINT_URL": "https://example",
                    "NEBIUS_S3_ACCESS_KEY_ID": "id",
                    "NEBIUS_S3_SECRET_ACCESS_KEY": "secret",
                },
                clear=False,
            ):
                # Avoid real sqlite checkpoint on fake bytes: patch connect
                with patch("datagen.preload_metadata.sqlite3.connect") as connect:
                    conn = MagicMock()
                    connect.return_value.__enter__.return_value = conn
                    upload_preload(local_dir, client=client)
            keys = [call.args[2] for call in client.upload_file.call_args_list]
            self.assertEqual(
                keys,
                ["arxiv-index/metadata.sqlite", "arxiv-index/PRELOAD_READY.json"],
            )


@unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss-cpu not installed")
class FaissTests(unittest.TestCase):
    def test_builds_loads_and_searches_exact_sample_index(self):
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory)
            connection = connect_database(index_dir)
            upsert_papers(
                connection,
                [
                    ArxivPaper("1", "Attention Paper", "Channel attention."),
                    ArxivPaper("2", "Geometry Paper", "Linear regions."),
                ],
            )
            encoder = FakeEncoder()
            embed_pending(connection, index_dir, encoder)
            manifest = build_faiss_index(connection, index_dir)
            connection.close()

            self.assertEqual(manifest["paper_count"], 2)
            with ArxivIndex(index_dir, encoder=encoder) as index:
                results = index.search(["attention mechanisms"], k=2)

            self.assertEqual(results[0][0]["arxiv_id"], "1")
            self.assertLessEqual(
                results[0][0]["distance"], results[0][1]["distance"]
            )


if __name__ == "__main__":
    unittest.main()
