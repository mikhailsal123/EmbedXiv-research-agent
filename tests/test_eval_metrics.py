"""Tests for eval funnel metrics."""

import unittest

from eval.metrics import compute_funnel


class FunnelMetricsTests(unittest.TestCase):
    def test_compute_funnel_counts_stages(self):
        payload = {
            "source": "papers/example.pdf",
            "source_arxiv_ids": ["1234.56789"],
            "run": {"label": "full", "elapsed_seconds": 120.0},
            "candidates": [
                {
                    "arxiv_id": "1111.11111",
                    "screen": {"decision": "read_full"},
                    "judgment": {
                        "decision": "keep",
                        "stage": "full_text",
                    },
                    "retrieval_source": "search_refinement",
                },
                {
                    "arxiv_id": "2222.22222",
                    "screen": {"decision": "read_full"},
                    "judgment": {
                        "decision": "drop",
                        "stage": "cap",
                    },
                },
                {
                    "arxiv_id": "3333.33333",
                    "screen": {"decision": "drop"},
                    "judgment": {
                        "decision": "drop",
                        "stage": "screen",
                    },
                },
                {
                    "arxiv_id": "4444.44444",
                    "screen": {"decision": "read_full"},
                    "judgment": {
                        "decision": "drop",
                        "stage": "full_text",
                    },
                },
            ],
        }
        metrics = compute_funnel(payload)
        self.assertEqual(metrics.total_candidates, 4)
        self.assertEqual(metrics.screen_read_full, 3)
        self.assertEqual(metrics.screen_drop, 1)
        self.assertEqual(metrics.full_text_keep, 1)
        self.assertEqual(metrics.cap_drop, 1)
        self.assertEqual(metrics.final_kept, 1)
        self.assertEqual(metrics.refinement_kept, 1)
        self.assertFalse(metrics.source_contamination)

    def test_source_contamination_detected(self):
        payload = {
            "source_arxiv_ids": ["1234.56789"],
            "candidates": [
                {
                    "arxiv_id": "1234.56789",
                    "judgment": {"decision": "keep", "stage": "full_text"},
                }
            ],
        }
        metrics = compute_funnel(payload)
        self.assertTrue(metrics.source_contamination)


if __name__ == "__main__":
    unittest.main()
