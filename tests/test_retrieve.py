import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_index import connect_database
from extract_claims import (
    ConceptualClaim,
    ImplementationDetail,
    ResearchProblem,
)
from retrieve import (
    build_queries,
    enrich_semantic_scholar,
    retrieve_candidates,
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
        self.calls.append((url, kwargs))
        return next(self.responses)


class QueryGenerationTests(unittest.TestCase):
    def test_builds_contextual_problem_claim_and_detail_queries(self):
        queries = build_queries(sample_problem())

        self.assertEqual(len(queries), 5)
        self.assertEqual(
            [query.level for query in queries],
            ["problem", "claim", "claim", "implementation", "implementation"],
        )
        self.assertEqual(
            [query.query_type for query in queries],
            ["direct", "direct", "functional", "direct", "alternative"],
        )
        self.assertIn("CNN feature maps contain", queries[0].query)
        self.assertIn("Problem:", queries[1].query)
        self.assertIn("Claim:", queries[1].query)
        self.assertEqual(
            queries[2].source_text,
            "Selects useful information at two granularities.",
        )
        self.assertIn("Desired implementation role:", queries[4].query)
        self.assertNotIn("Implementation detail:", queries[4].query)

    def test_deduplication_preserves_vector_matches(self):
        paper = {
            "arxiv_id": "2001.01072v2",
            "title": "Linear Regions",
            "abstract": "Abstract",
            "distance": 0.2,
            "rank": 1,
        }
        index = FakeVectorIndex([[paper] for _ in range(5)])

        candidates = retrieve_candidates(sample_problem(), index)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["arxiv_id"], "2001.01072")
        self.assertEqual(len(candidates[0]["matched_queries"]), 5)
        self.assertEqual(len(index.queries), 5)


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

        call = session.calls[0][1]
        self.assertEqual(call["headers"], {"x-api-key": "secret"})
        self.assertEqual(call["json"], {"ids": ["ARXIV:2001.01072"]})
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

    @patch("retrieve.time.sleep")
    @patch("retrieve.random.uniform", return_value=0)
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


if __name__ == "__main__":
    unittest.main()
