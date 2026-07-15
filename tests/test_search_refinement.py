"""Tests for bounded search refinement."""

import unittest
from types import SimpleNamespace

from search_refinement import (
    RefinementSearchQuery,
    SearchRefinementAssessment,
    SearchRefinementPlan,
    run_search_refinement,
)
from extract_claims import ConceptualClaim, ImplementationDetail, ResearchProblem


def sample_problem() -> ResearchProblem:
    return ResearchProblem(
        problem="CNN feature maps mix useful and irrelevant information.",
        domain="computer vision",
        keywords=["attention", "feature refinement", "CNN"],
        claims=[
            ConceptualClaim(
                claim="Sequential channel and spatial selection improves features.",
                functional_role="Selects useful information at two granularities.",
                implementation_details=[
                    ImplementationDetail(
                        detail="Apply channel attention followed by spatial attention.",
                        functional_role="Refines feature maps by selecting useful signals.",
                    )
                ],
            )
        ],
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


class FakeVectorIndex:
    def __init__(self, result_sets_by_call):
        self.result_sets_by_call = list(result_sets_by_call)
        self.calls = []

    def search(self, query_texts, k):
        self.calls.append((list(query_texts), k))
        if not self.result_sets_by_call:
            raise AssertionError("Unexpected extra search call")
        return self.result_sets_by_call.pop(0)


class SearchRefinementTests(unittest.TestCase):
    def test_search_refinement_merges_metadata_and_excludes_ids(self):
        problem = sample_problem()
        plan = SearchRefinementPlan(
            queries=[
                RefinementSearchQuery(
                    target_id="p0.c0.d0",
                    query="coordinate attention feature refinement",
                    rationale="Different positional attention mechanism.",
                )
            ]
        )
        client = FakeClient(
            [
                plan,
                SearchRefinementAssessment(
                    target_id="p0.c0.d0",
                    found_plausible_substitutes=True,
                    diagnosis="Found a plausible substitute.",
                    followup_queries=[],
                ),
            ]
        )
        index = FakeVectorIndex(
            [
                [
                    [
                        {
                            "arxiv_id": "2103.02907",
                            "title": "Coordinate Attention",
                            "abstract": "Encodes positional information.",
                            "distance": 0.1,
                            "rank": 1,
                        },
                        {
                            "arxiv_id": "1807.06521",
                            "title": "Source Paper",
                            "abstract": "Excluded.",
                            "distance": 0.2,
                            "rank": 2,
                        },
                    ]
                ]
            ]
        )

        candidates, trace = run_search_refinement(
            [problem],
            index,
            client=client,
            max_rounds=2,
            exclude_ids={"1807.06521"},
        )

        self.assertEqual([candidate["arxiv_id"] for candidate in candidates], ["2103.02907"])
        self.assertEqual(candidates[0]["retrieval_source"], "search_refinement")
        self.assertEqual(
            candidates[0]["matched_queries"][0]["query_type"],
            "refined_substitute",
        )
        self.assertEqual(trace[0]["candidate_count"], 1)

    def test_refinement_rewrites_once_and_stops_at_max_rounds(self):
        problem = sample_problem()
        client = FakeClient(
            [
                SearchRefinementPlan(
                    queries=[
                        RefinementSearchQuery(
                            target_id="p0.c0.d0",
                            query="feature refinement attention alternatives",
                            rationale="Initial broad substitute search.",
                        )
                    ]
                ),
                SearchRefinementAssessment(
                    target_id="p0.c0.d0",
                    found_plausible_substitutes=False,
                    diagnosis="Too generic.",
                    followup_queries=[
                        "coordinate attention positional feature refinement"
                    ],
                ),
                SearchRefinementAssessment(
                    target_id="p0.c0.d0",
                    found_plausible_substitutes=False,
                    diagnosis="Still weak.",
                    followup_queries=["dynamic convolution feature selection"],
                ),
            ]
        )
        index = FakeVectorIndex(
            [
                [
                    [
                        {
                            "arxiv_id": "1000.00001",
                            "title": "Generic Attention",
                            "abstract": "Generic attention.",
                            "distance": 0.4,
                            "rank": 1,
                        }
                    ]
                ],
                [
                    [
                        {
                            "arxiv_id": "2103.02907",
                            "title": "Coordinate Attention",
                            "abstract": "Alternative feature refinement.",
                            "distance": 0.2,
                            "rank": 1,
                        }
                    ]
                ],
            ]
        )

        candidates, trace = run_search_refinement(
            [problem],
            index,
            client=client,
            max_rounds=2,
            max_followups_per_target=1,
        )

        self.assertEqual(len(index.calls), 2)
        self.assertEqual(index.calls[0][0], ["feature refinement attention alternatives"])
        self.assertEqual(
            index.calls[1][0],
            ["coordinate attention positional feature refinement"],
        )
        self.assertEqual(len(trace), 2)
        self.assertEqual(len(candidates), 2)

    def test_max_rounds_one_never_runs_followup(self):
        problem = sample_problem()
        client = FakeClient(
            [
                SearchRefinementPlan(
                    queries=[
                        RefinementSearchQuery(
                            target_id="p0.c0",
                            query="alternative representation constraints",
                            rationale="Claim-level substitute search.",
                        )
                    ]
                ),
                SearchRefinementAssessment(
                    target_id="p0.c0",
                    found_plausible_substitutes=False,
                    diagnosis="No substitute yet.",
                    followup_queries=["auxiliary structure constraints"],
                ),
            ]
        )
        index = FakeVectorIndex([[[]]])

        _candidates, trace = run_search_refinement(
            [problem],
            index,
            client=client,
            max_rounds=1,
        )

        self.assertEqual(len(index.calls), 1)
        self.assertEqual(len(trace), 1)


if __name__ == "__main__":
    unittest.main()
