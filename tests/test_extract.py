import unittest
from types import SimpleNamespace

from pydantic import ValidationError

from extract_claims import (
    ConceptualClaim,
    ExtractionResult,
    ImplementationDetail,
    ResearchProblem,
    extract_claims,
)


class FakeCompletions:
    def __init__(self, result):
        self.result = result
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        message = SimpleNamespace(parsed=self.result)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, result):
        self.completions = FakeCompletions(result)
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )


class ExtractionTests(unittest.TestCase):
    def test_returns_nested_problem_structure(self):
        expected = ExtractionResult(
            problems=[
                ResearchProblem(
                    problem="Feature maps contain irrelevant information.",
                    domain="computer vision",
                    keywords=["attention", "feature refinement", "CNN"],
                    claims=[
                        ConceptualClaim(
                            claim="Selective attention improves representations.",
                            functional_role="Suppresses irrelevant information.",
                            implementation_details=[
                                ImplementationDetail(
                                    detail="Apply channel attention.",
                                    functional_role="Selects useful feature groups.",
                                )
                            ],
                        )
                    ],
                )
            ]
        )
        client = FakeClient(expected)

        problems = extract_claims("A non-empty abstract.", client=client)

        self.assertEqual(problems, expected.problems)
        self.assertIs(
            client.completions.call["response_format"],
            ExtractionResult,
        )

    def test_rejects_empty_input_before_calling_model(self):
        with self.assertRaises(ValueError):
            extract_claims("   ", client=FakeClient(None))

    def test_schema_rejects_invalid_counts_and_empty_strings(self):
        with self.assertRaises(ValidationError):
            ResearchProblem(
                problem="A specific limitation.",
                domain="machine learning",
                keywords=["only", "two"],
                claims=[],
            )

        with self.assertRaises(ValidationError):
            ImplementationDetail(
                detail=" ",
                functional_role="A valid role.",
            )

    def test_schema_rejects_extra_fields_and_duplicates(self):
        with self.assertRaises(ValidationError):
            ImplementationDetail.model_validate(
                {
                    "detail": "Use channel attention.",
                    "functional_role": "Select feature groups.",
                    "unexpected": True,
                }
            )

        claim = ConceptualClaim(
            claim="Selective attention improves representations.",
            functional_role="Suppresses irrelevant information.",
        )
        with self.assertRaises(ValidationError):
            ResearchProblem(
                problem="Feature maps contain irrelevant information.",
                domain="computer vision",
                keywords=["attention", "Attention", "CNN"],
                claims=[claim],
            )


if __name__ == "__main__":
    unittest.main()
