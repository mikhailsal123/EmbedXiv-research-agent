"""Tests for SPECTER2 query mapping, vector candidate search, and S2 enrichment."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extract_claims import (
    ConceptualClaim,
    ImplementationDetail,
    ResearchProblem,
)
from search_candidates import (
    build_queries,
    connect_database,
    detect_source_arxiv_id,
    enrich_semantic_scholar,
    open_index,
    search_candidates,
    specter2_query_text,
)


def sample_problem() -> ResearchProblem:
    return ResearchProblem(
        problem="CNN feature maps contain useful and irrelevant information.",
        domain="computer vision",
        keywords=["attention", "feature refinement", "CNN"],
        claims=[
            ConceptualClaim(
                claim="Channel and spatial selection improves representations.",
                functional_role="Selects useful information at two granularities.",
                implementation_details=[
                    ImplementationDetail(
                        detail="Apply channel attention followed by spatial attention.",
                        functional_role="Refines one dimension before another.",
                    )
                ],
            )
        ],
    )


class FakeVectorIndex:
    def __init__(self, result_sets):
        self.result_sets = result_sets
        self.queries = None

    def search(self, query_texts, k):
        self.queries = list(query_texts)
        return self.result_sets


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return next(self.responses)

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return next(self.responses)


class QueryGenerationTests(unittest.TestCase):
    def test_specter2_query_text_joins_raw_sentences(self):
        self.assertEqual(
            specter2_query_text("First idea", "computer vision", "a, b"),
            "First idea. computer vision. a, b.",
        )

    def test_builds_specter2_style_problem_claim_and_detail_queries(self):
        problem = sample_problem()
        queries = build_queries(problem)

        self.assertEqual(len(queries), 5)
        self.assertEqual(
            [query.level for query in queries],
            ["problem", "claim", "claim", "implementation", "implementation"],
        )
        self.assertEqual(
            [query.query_type for query in queries],
            ["direct", "direct", "functional", "direct", "alternative"],
        )
        self.assertEqual(
            queries[0].query,
            specter2_query_text(
                problem.problem,
                problem.domain,
                ", ".join(problem.keywords),
            ),
        )
        self.assertEqual(
            queries[1].query,
            "Channel and spatial selection improves representations.",
        )
        self.assertEqual(
            queries[2].query,
            "Selects useful information at two granularities.",
        )
        self.assertEqual(
            queries[3].query,
            "Apply channel attention followed by spatial attention.",
        )
        self.assertEqual(
            queries[4].query,
            "Refines one dimension before another.",
        )
        self.assertNotIn("Problem:", queries[0].query)
        self.assertNotIn("Claim:", queries[1].query)

    def test_deduplication_preserves_vector_matches(self):
        paper = {
            "arxiv_id": "2001.01072v2",
            "title": "Linear Regions",
            "abstract": "Abstract",
            "distance": 0.2,
            "rank": 1,
        }
        index = FakeVectorIndex([[paper] for _ in range(5)])

        candidates = search_candidates(sample_problem(), index)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["arxiv_id"], "2001.01072")
        self.assertEqual(len(candidates[0]["matched_queries"]), 5)
        self.assertEqual(len(index.queries), 5)

    def test_excludes_source_arxiv_id_from_vector_hits(self):
        hits = [
            {
                "arxiv_id": "1807.06521",
                "title": "CBAM itself",
                "abstract": "Self",
                "distance": 0.01,
                "rank": 1,
            },
            {
                "arxiv_id": "1807.06514",
                "title": "BAM",
                "abstract": "Sibling",
                "distance": 0.2,
                "rank": 2,
            },
        ]
        index = FakeVectorIndex([[hits[0], hits[1]] for _ in range(5)])

        candidates = search_candidates(
            sample_problem(),
            index,
            exclude_ids={"1807.06521"},
        )

        self.assertEqual([c["arxiv_id"] for c in candidates], ["1807.06514"])

    def test_detect_source_arxiv_id_from_header_not_bibliography(self):
        text = (
            "arXiv:1807.06521v1 [cs.CV] 18 Jul 2018\n"
            "CBAM: Convolutional Block Attention Module\n\n"
            + ("body " * 500)
            + "\nReferences\n[1] arXiv:1709.01507 Squeeze-and-Excitation\n"
        )
        self.assertEqual(detect_source_arxiv_id(text), "1807.06521")

    @patch.dict("os.environ", {"DATABASE_URL": "postgresql://example"}, clear=False)
    @patch("search_candidates.PgvectorIndex")
    def test_open_index_uses_postgres_when_database_url_set(self, mock_index):
        mock_index.return_value.__enter__.return_value = object()
        with open_index():
            pass
        mock_index.assert_called_once()


class SemanticScholarEnrichmentTests(unittest.TestCase):
    def test_batch_enrichment_uses_api_key_and_preserves_arxiv_fields(self):
        payload = [
            {
                "paperId": "s2-paper",
                "title": "Published title",
                "citationCount": 42,
            }
        ]
        session = FakeSession([FakeResponse(payload=payload)])
        candidates = [
            {
                "arxiv_id": "2001.01072",
                "title": "Authoritative arXiv title",
            }
        ]

        enrich_semantic_scholar(
            candidates,
            api_key="secret",
            session=session,
            request_delay=0,
        )

        call = session.calls[0]
        self.assertEqual(call[0], "POST")
        self.assertEqual(call[2]["headers"], {"x-api-key": "secret"})
        self.assertEqual(call[2]["json"], {"ids": ["ARXIV:2001.01072"]})
        self.assertEqual(candidates[0]["title"], "Authoritative arXiv title")
        self.assertEqual(candidates[0]["paperId"], "s2-paper")
        self.assertEqual(candidates[0]["citationCount"], 42)
        self.assertEqual(candidates[0]["semantic_scholar"]["status"], "ok")

    def test_missing_and_unavailable_enrichment_never_drop_candidates(self):
        candidates = [{"arxiv_id": "9999.99999", "title": "Local result"}]
        session = FakeSession([FakeResponse(payload=[None])])

        result = enrich_semantic_scholar(
            candidates,
            api_key="secret",
            session=session,
            request_delay=0,
        )

        self.assertEqual(result[0]["title"], "Local result")
        self.assertEqual(result[0]["semantic_scholar"]["status"], "not_found")

    @patch("search_candidates.time.sleep")
    @patch("search_candidates.random.uniform", return_value=0)
    def test_retries_transient_batch_failure(self, _random, sleep):
        session = FakeSession(
            [
                FakeResponse(status_code=429, headers={"Retry-After": "0"}),
                FakeResponse(payload=[None]),
            ]
        )
        candidates = [{"arxiv_id": "9999.99999", "title": "Local result"}]

        enrich_semantic_scholar(
            candidates,
            api_key="secret",
            session=session,
            request_delay=0,
        )

        self.assertEqual(len(session.calls), 2)
        sleep.assert_called_once_with(0)

    def test_caches_success_and_negative_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_database(Path(directory))
            session = FakeSession([FakeResponse(payload=[None])])
            candidates = [{"arxiv_id": "9999.99999", "title": "Local result"}]

            enrich_semantic_scholar(
                candidates,
                api_key="secret",
                session=session,
                cache_connection=connection,
                request_delay=0,
            )
            second_session = FakeSession([])
            second = [{"arxiv_id": "9999.99999", "title": "Local result"}]
            enrich_semantic_scholar(
                second,
                api_key="secret",
                session=second_session,
                cache_connection=connection,
                request_delay=0,
            )

            self.assertEqual(second_session.calls, [])
            self.assertEqual(second[0]["semantic_scholar"]["status"], "not_found")
            connection.close()


class SemanticScholarRecommendTests(unittest.TestCase):
    def test_recommendations_keep_arxiv_papers_and_skip_duplicates(self):
        from search_candidates import recommend_semantic_scholar

        payload = {
            "recommendedPapers": [
                {
                    "paperId": "s2-a",
                    "title": "Related A",
                    "abstract": "About scarce labels.",
                    "externalIds": {"ArXiv": "2301.11111"},
                },
                {
                    "paperId": "s2-b",
                    "title": "Journal only",
                    "abstract": "No arxiv id.",
                    "externalIds": {"DOI": "10.1/x"},
                },
                {
                    "paperId": "s2-c",
                    "title": "Already seen",
                    "abstract": "Dup.",
                    "externalIds": {"ArXiv": "2001.01072"},
                },
            ]
        }
        session = FakeSession([FakeResponse(payload=payload)])
        seeds = [{"arxiv_id": "2001.01072", "title": "Seed"}]

        recs = recommend_semantic_scholar(
            seeds,
            api_key="secret",
            session=session,
            limit_per_seed=5,
            request_delay=0,
            exclude_ids={"2001.01072"},
        )

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["arxiv_id"], "2301.11111")
        self.assertEqual(recs[0]["retrieval_source"], "semantic_scholar_recommend")
        self.assertEqual(recs[0]["recommended_from"], "2001.01072")
        self.assertEqual(session.calls[0][0], "GET")
        self.assertIn("ARXIV:2001.01072", session.calls[0][1])


if __name__ == "__main__":
    unittest.main()
