"""Tests for two-stage Qwen candidate judging."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from extract_claims import ConceptualClaim, ResearchProblem
from judge_candidates import (
    CandidateJudgment,
    ScreenJudgment,
    ScreenQueryResult,
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


if __name__ == "__main__":
    unittest.main()
