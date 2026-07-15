"""Tests for two-stage Qwen candidate judging."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from extract_claims import ConceptualClaim, ResearchProblem
from judge_candidates import (
    CandidateJudgment,
    NodeTopSelection,
    ScreenJudgment,
    ScreenQueryResult,
    cap_kept_candidates_per_node,
    judge_candidates,
    kept_candidates,
)


class FakeCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("Unexpected extra model call")
        result = self.results.pop(0)
        message = SimpleNamespace(parsed=result)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, results):
        self.completions = FakeCompletions(results)
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.problems = [
            ResearchProblem(
                problem="Models overfit when supervision is scarce.",
                domain="machine learning",
                keywords=["generalization", "overfitting", "scarce labels"],
                claims=[
                    ConceptualClaim(
                        claim="Auxiliary structure improves generalization.",
                        functional_role="Adds constraints that reduce overfitting.",
                        implementation_details=[],
                    )
                ],
            )
        ]
        self.candidates = [
            {
                "arxiv_id": "2301.00001",
                "title": "Useful Paper",
                "abstract": "We address scarce-label overfitting with structure.",
                "best_distance": 0.2,
                "best_rank": 1,
                "matched_queries": [
                    {
                        "level": "problem",
                        "query_type": "direct",
                        "source_text": self.problems[0].problem,
                    }
                ],
            },
            {
                "arxiv_id": "2301.00002",
                "title": "Unrelated Paper",
                "abstract": "A survey of tropical fruit classification.",
                "best_distance": 0.9,
                "best_rank": 2,
                "matched_queries": [],
            },
        ]

    def test_screen_drops_then_full_text_keeps_survivor(self):
        screen = ScreenQueryResult(
            judgments=[
                ScreenJudgment(
                    arxiv_id="2301.00001",
                    decision="read_full",
                    why="Same scarce-label gap.",
                ),
            ]
        )
        unmatched_screen = ScreenQueryResult(
            judgments=[
                ScreenJudgment(
                    arxiv_id="2301.00002",
                    decision="drop",
                    why="Unrelated domain.",
                ),
            ]
        )
        full = CandidateJudgment(
            arxiv_id="2301.00001",
            decision="keep",
            relation="same_problem",
            why="Full text confirms the same problem framing.",
            primary_level="problem",
        )
        client = FakeClient([screen, unmatched_screen, full])
        judged = judge_candidates(
            self.problems,
            self.candidates,
            client=client,
            fetch_pdfs=True,
            pdf_text_by_id={"2301.00001": "Full paper body about scarce labels."},
            request_delay=0.0,
        )
        self.assertEqual(len(client.completions.calls), 3)
        self.assertEqual(judged[0]["screen"]["decision"], "read_full")
        self.assertEqual(judged[0]["judgment"]["decision"], "keep")
        self.assertEqual(judged[0]["judgment"]["stage"], "full_text")
        self.assertEqual(judged[1]["judgment"]["decision"], "drop")
        self.assertEqual(judged[1]["judgment"]["stage"], "screen")
        self.assertEqual(len(kept_candidates(judged)), 1)

    def test_missing_screen_entry_is_dropped(self):
        screen = ScreenQueryResult(
            judgments=[
                ScreenJudgment(
                    arxiv_id="2301.00002",
                    decision="drop",
                    why="Different group.",
                )
            ]
        )
        unmatched_screen = ScreenQueryResult(
            judgments=[
                ScreenJudgment(
                    arxiv_id="2301.00002",
                    decision="drop",
                    why="Weak.",
                )
            ]
        )
        judged = judge_candidates(
            self.problems,
            self.candidates,
            client=FakeClient([screen, unmatched_screen]),
            fetch_pdfs=False,
        )
        self.assertEqual(judged[0]["judgment"]["decision"], "drop")


class CapKeptPerNodeTests(unittest.TestCase):
    def setUp(self):
        self.problems = [
            ResearchProblem(
                problem="Models overfit when supervision is scarce.",
                domain="machine learning",
                keywords=["generalization", "overfitting", "scarce labels"],
                claims=[
                    ConceptualClaim(
                        claim="Auxiliary structure improves generalization.",
                        functional_role="Adds constraints that reduce overfitting.",
                        implementation_details=[],
                    )
                ],
            )
        ]

    def _kept_candidate(self, arxiv_id: str, distance: float) -> dict:
        return {
            "arxiv_id": arxiv_id,
            "title": f"Paper {arxiv_id}",
            "abstract": "Relevant scarce-label generalization work.",
            "best_distance": distance,
            "best_rank": 1,
            "matched_queries": [
                {
                    "level": "problem",
                    "query_type": "direct",
                    "source_text": self.problems[0].problem,
                    "problem_index": 0,
                    "distance": distance,
                    "rank": 1,
                }
            ],
            "judgment": CandidateJudgment(
                arxiv_id=arxiv_id,
                decision="keep",
                relation="same_problem",
                why=f"Useful for {arxiv_id}.",
                primary_level="problem",
            ).model_dump(),
        }

    def test_cap_keeps_all_when_under_limit(self):
        candidates = [self._kept_candidate(f"2301.0000{i}", 0.1 * i) for i in range(3)]
        capped = cap_kept_candidates_per_node(self.problems, candidates, max_per_node=5)
        self.assertEqual(len(kept_candidates(capped)), 3)
        self.assertTrue(all(candidate.get("attachment") for candidate in kept_candidates(capped)))

    def test_cap_limits_excess_kept_for_one_node(self):
        candidates = [self._kept_candidate(f"2301.0000{i}", 0.1 * i) for i in range(7)]
        selection = NodeTopSelection(
            selected_arxiv_ids=[
                "2301.00000",
                "2301.00001",
                "2301.00002",
            ]
        )
        capped = cap_kept_candidates_per_node(
            self.problems,
            candidates,
            client=FakeClient([selection]),
            max_per_node=5,
        )
        kept = kept_candidates(capped)
        self.assertEqual(len(kept), 3)
        self.assertEqual(
            {candidate["arxiv_id"] for candidate in kept},
            {"2301.00000", "2301.00001", "2301.00002"},
        )
        dropped = [
            candidate
            for candidate in capped
            if candidate["judgment"]["decision"] == "drop"
        ]
        self.assertEqual(len(dropped), 4)
        self.assertTrue(all(item["judgment"]["stage"] == "cap" for item in dropped))

    def test_cap_assigns_paper_to_single_strongest_node(self):
        candidate = self._kept_candidate("2301.00009", 0.2)
        candidate["matched_queries"] = [
            {
                "level": "problem",
                "query_type": "direct",
                "source_text": self.problems[0].problem,
                "problem_index": 0,
                "distance": 0.8,
                "rank": 3,
            },
            {
                "level": "claim",
                "query_type": "direct",
                "source_text": self.problems[0].claims[0].claim,
                "problem_index": 0,
                "claim_index": 0,
                "distance": 0.2,
                "rank": 1,
            },
        ]
        candidate["judgment"]["primary_level"] = "claim"
        capped = cap_kept_candidates_per_node(
            self.problems,
            [candidate],
            max_per_node=5,
        )
        kept = kept_candidates(capped)[0]
        self.assertEqual(kept["attachment"]["node_path"], "0.0")
        self.assertEqual(kept["attachment"]["node_kind"], "claim")


if __name__ == "__main__":
    unittest.main()
