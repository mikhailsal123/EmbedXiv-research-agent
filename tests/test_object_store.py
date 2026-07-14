"""Tests for loading the published corpus from Nebius Object Storage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from search_candidates import download_generation, latest_generation


class ObjectStoreTests(unittest.TestCase):
    def test_latest_generation_reads_bucket_pointer(self):
        client = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps({"generation": "20260713T120000Z"}).encode()
        client.get_object.return_value = {"Body": body}

        with patch.dict(
            "os.environ",
            {
                "NEBIUS_S3_BUCKET": "embedxiv-bucket",
                "NEBIUS_INDEX_PREFIX": "arxiv-index",
            },
            clear=False,
        ):
            generation = latest_generation(client=client)

        self.assertEqual(generation, "20260713T120000Z")
        client.get_object.assert_called_once_with(
            Bucket="embedxiv-bucket",
            Key="arxiv-index/LATEST.json",
        )

    def test_download_generation_fetches_search_artifacts(self):
        client = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ",
                {
                    "NEBIUS_S3_BUCKET": "embedxiv-bucket",
                    "NEBIUS_INDEX_PREFIX": "arxiv-index",
                },
                clear=False,
            ):
                root = download_generation(
                    Path(directory),
                    generation="20260713T120000Z",
                    client=client,
                )

            self.assertEqual(root, Path(directory) / "20260713T120000Z")
            keys = [call.args[1] for call in client.download_file.call_args_list]
            self.assertEqual(
                keys,
                [
                    "arxiv-index/generations/20260713T120000Z/manifest.json",
                    "arxiv-index/generations/20260713T120000Z/index.faiss",
                    "arxiv-index/generations/20260713T120000Z/metadata.sqlite",
                ],
            )


if __name__ == "__main__":
    unittest.main()
