import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arxiv_index import (
    DIMENSION,
    ArxivIndex,
    ArxivPaper,
    build_faiss_index,
    canonical_arxiv_id,
    connect_database,
    embed_pending,
    iter_snapshot,
    parse_oai_page,
    upsert_papers,
)


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


class MetadataTests(unittest.TestCase):
    def test_canonicalizes_new_and_legacy_ids_without_corrupting_archive(self):
        self.assertEqual(canonical_arxiv_id("ARXIV:2001.01072v3"), "2001.01072")
        self.assertEqual(
            canonical_arxiv_id("oai:arXiv.org:solv-int/9701001v2"),
            "solv-int/9701001",
        )

    def test_parses_oai_records_and_deletions(self):
        payload = b"""<?xml version="1.0"?>
        <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
                 xmlns:arXiv="http://arxiv.org/OAI/arXiv/">
          <ListRecords>
            <record>
              <header>
                <identifier>oai:arXiv.org:2001.01072</identifier>
                <datestamp>2020-01-05</datestamp>
              </header>
              <metadata>
                <arXiv:arXiv>
                  <arXiv:id>2001.01072</arXiv:id>
                  <arXiv:title> Linear Regions </arXiv:title>
                  <arXiv:abstract> A geometric study. </arXiv:abstract>
                  <arXiv:categories>cs.LG</arXiv:categories>
                  <arXiv:authors>
                    <arXiv:author>
                      <arXiv:keyname>Zhang</arXiv:keyname>
                      <arXiv:forenames>Xiao</arXiv:forenames>
                    </arXiv:author>
                  </arXiv:authors>
                </arXiv:arXiv>
              </metadata>
            </record>
            <record>
              <header status="deleted">
                <identifier>oai:arXiv.org:9999.99999</identifier>
                <datestamp>2020-01-06</datestamp>
              </header>
            </record>
            <resumptionToken>next-page</resumptionToken>
          </ListRecords>
        </OAI-PMH>"""

        papers, token = parse_oai_page(payload)

        self.assertEqual(token, "next-page")
        self.assertEqual(papers[0].title, " Linear Regions ")
        self.assertEqual(papers[0].authors, "Xiao Zhang")
        self.assertTrue(papers[1].deleted)

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
            self.assertEqual(
                embed_pending(connection, index_dir, encoder),
                0,
            )
            upsert_papers(
                connection,
                [ArxivPaper("2", "Attention Geometry", "Updated abstract.")],
            )
            self.assertEqual(
                embed_pending(connection, index_dir, encoder),
                1,
            )
            connection.close()

    def test_snapshot_allows_missing_abstract_but_rejects_missing_id(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "metadata.json"
            snapshot.write_text(json.dumps({"id": "1", "title": "Title"}) + "\n")
            paper = next(iter_snapshot(snapshot))
            self.assertEqual(paper.abstract, "")

            snapshot.write_text(json.dumps({"title": "Missing ID"}) + "\n")
            with self.assertRaises(KeyError):
                next(iter_snapshot(snapshot))


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
