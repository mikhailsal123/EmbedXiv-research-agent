"""Tests for suggestion card building and rendering."""

import tempfile
import unittest
from pathlib import Path

from suggestion_cards import (
    build_suggestion_cards,
    group_cards,
    render_html,
    render_markdown,
    write_suggestion_outputs,
)


class SuggestionCardTests(unittest.TestCase):
    def setUp(self):
        self.problems = [
            {
                "problem": "Scarce labels cause overfitting.",
                "domain": "machine learning",
                "keywords": ["scarce labels", "regularization", "overfitting"],
                "claims": [
                    {
                        "claim": "Regularization reduces scarce-label overfitting.",
                        "functional_role": "Stabilize learning under weak supervision.",
                        "implementation_details": [
                            {
                                "detail": "Add an entropy penalty on predictions.",
                                "functional_role": "Discourage overconfident outputs.",
                            },
                            {
                                "detail": "Unused alternate loss that never matched.",
                                "functional_role": "Should be filtered out.",
                            },
                        ],
                    },
                    {
                        "claim": "Unmatched claim that should be omitted.",
                        "functional_role": "No kept hits.",
                        "implementation_details": [
                            {
                                "detail": "Also unmatched.",
                                "functional_role": "No kept hits.",
                            }
                        ],
                    },
                ],
            }
        ]
        self.candidates = [
            {
                "arxiv_id": "2301.00002",
                "title": "Unrelated dropped",
                "abstract": "Fruit classification.",
                "url": "https://arxiv.org/abs/2301.00002",
                "best_distance": 0.1,
                "datestamp": "2023-01-02",
                "judgment": {
                    "decision": "drop",
                    "relation": "irrelevant",
                    "why": "Wrong domain.",
                    "primary_level": "problem",
                },
            },
            {
                "arxiv_id": "2301.00001",
                "title": "Scarce Label Regularization",
                "abstract": "We reduce overfitting under scarce supervision.",
                "url": "https://arxiv.org/abs/2301.00001",
                "best_distance": 0.4,
                "citationCount": 12,
                "datestamp": "2023-01-01",
                "matched_queries": [
                    {
                        "level": "problem",
                        "query_type": "direct",
                        "problem_index": 0,
                        "claim_index": None,
                        "detail_index": None,
                        "distance": 0.4,
                    }
                ],
                "judgment": {
                    "decision": "keep",
                    "relation": "same_problem",
                    "why": "Same scarce-label gap.",
                    "primary_level": "problem",
                    "stage": "full_text",
                },
            },
            {
                "arxiv_id": "2301.00003",
                "title": "Alternate Mechanism",
                "abstract": "A different training objective for the same role.",
                "best_distance": 0.5,
                "year": 2022,
                "retrieval_source": "semantic_scholar_recommend",
                "recommended_from": "2301.00001",
                "matched_queries": [
                    {
                        "level": "implementation",
                        "query_type": "direct",
                        "problem_index": 0,
                        "claim_index": 0,
                        "detail_index": 0,
                        "distance": 0.5,
                    }
                ],
                "judgment": {
                    "decision": "keep",
                    "relation": "implementation_alternative",
                    "why": "Different mechanism, same role.",
                    "primary_level": "implementation",
                    "stage": "full_text",
                },
            },
        ]

    def test_flat_sibling_boxes(self):
        cards = build_suggestion_cards(self.candidates, problems=self.problems)
        boxes = group_cards(cards, problems=self.problems)
        # Problem (with cards) + Claim (illustration, no cards) + Implementation
        self.assertEqual([box["kind"] for box in boxes], [
            "problem",
            "claim",
            "implementation",
        ])
        self.assertEqual(boxes[0]["text"], "Scarce labels cause overfitting.")
        self.assertEqual(len(boxes[0]["cards"]), 1)
        self.assertEqual(
            boxes[1]["text"], "Regularization reduces scarce-label overfitting."
        )
        self.assertEqual(boxes[1]["cards"], [])
        self.assertEqual(
            boxes[2]["text"], "Add an entropy penalty on predictions."
        )
        self.assertEqual(len(boxes[2]["cards"]), 1)

    def test_renders_markdown_and_html(self):
        cards = build_suggestion_cards(self.candidates, problems=self.problems)
        md = render_markdown(cards, source="paper.pdf", problems=self.problems)
        page = render_html(cards, source="paper.pdf", problems=self.problems)
        self.assertIn("## Problem", md)
        self.assertIn("## Claim", md)
        self.assertIn("## Implementation", md)
        self.assertIn("box-problem", page)
        self.assertIn("box-claim", page)
        self.assertIn("box-implementation", page)
        self.assertIn("card-date", page)
        self.assertIn("2023-01-01", page)
        self.assertNotIn("Unmatched claim that should be omitted.", page)
        # Sibling sections, not claim nested inside problem markup
        problem_block = page.split('class="box box-problem"', 1)[1].split(
            'class="box box-claim"', 1
        )[0]
        self.assertNotIn("box-claim", problem_block)
        self.assertNotIn("box-implementation", problem_block)

    def test_writes_sidecar_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "results.json"
            written = write_suggestion_outputs(
                self.candidates,
                base,
                source="demo.pdf",
                problems=self.problems,
            )
            self.assertTrue(written["json"].is_file())
            self.assertTrue(written["markdown"].is_file())
            self.assertTrue(written["html"].is_file())
            self.assertEqual(len(written["cards"]), 2)
            self.assertEqual(len(written["groups"]), 3)


if __name__ == "__main__":
    unittest.main()
