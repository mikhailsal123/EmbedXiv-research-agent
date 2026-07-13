"""
Step 1: Extract problems, conceptual claims, and implementation details into a schema.

A problem states the paper's main research subject. Claims state the new ideas 
used to address it. Implementation details describe how each claim is realized.

Set env vars first:
  NEBIUS_ENDPOINT_URL
  NEBIUS_ENDPOINT_TOKEN
"""
import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from openai import OpenAI


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_local_env()

DEFAULT_EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "qwen3:32b")


def get_client() -> OpenAI:
    endpoint_url = os.getenv("NEBIUS_ENDPOINT_URL")
    endpoint_token = os.getenv("NEBIUS_ENDPOINT_TOKEN")
    if not endpoint_url or not endpoint_token:
        raise RuntimeError(
            "Set NEBIUS_ENDPOINT_URL and NEBIUS_ENDPOINT_TOKEN in .env"
        )
    return OpenAI(
        base_url=f"{endpoint_url.rstrip('/')}/v1",
        api_key=endpoint_token,
    )


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _duplicate_values(values: list[str]) -> list[str]:
    normalized = [value.casefold() for value in values]
    return sorted(
        {value for value in normalized if normalized.count(value) > 1}
    )


class ImplementationDetail(SchemaModel):
    detail: str = Field(
        min_length=1,
        description=(
            "A concrete mechanism, architecture choice, algorithm, objective, "
            "or training procedure used to implement the parent claim."
        )
    )
    functional_role: str = Field(
        min_length=1,
        description=(
            "Domain-neutral description of what this specific detail "
            "accomplishes, with domain-specific vocabulary removed."
        )
    )
class ConceptualClaim(SchemaModel):
    claim: str = Field(
        min_length=1,
        description=(
            "One sentence stating the paper's new conceptual idea, without "
            "implementation details."
        )
    )
    functional_role: str = Field(
        min_length=1,
        description=(
            "One sentence describing what goal the claim serves "
            "or what general pattern it represents, with ALL domain-specific "
            "vocabulary removed. Must be understandable to someone outside "
            "this field."
        )
    )
    implementation_details: list[ImplementationDetail] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Concrete technical choices used to realize this claim. Excludes "
            "results, benefits, observations, ablations, and validation."
        ),
    )

    @model_validator(mode="after")
    def implementation_details_must_be_unique(self) -> "ConceptualClaim":
        duplicates = _duplicate_values(
            [detail.detail for detail in self.implementation_details]
        )
        if duplicates:
            raise ValueError(
                f"duplicate implementation details: {', '.join(duplicates)}"
            )
        return self


class ResearchProblem(SchemaModel):
    problem: str = Field(
        min_length=1,
        description="One clear sentence stating the limitation or need addressed"
    )
    domain: str = Field(
        min_length=1,
        description="Specific field/subfield, e.g. 'computational chemistry'"
    )
    keywords: list[str] = Field(
        min_length=3,
        max_length=6,
        description="3-6 short search labels covering the problem and its claims"
    )
    claims: list[ConceptualClaim] = Field(
        min_length=1,
        max_length=4,
        description="Independent conceptual assertions proposed to address the problem"
    )

    @field_validator("keywords")
    @classmethod
    def keywords_must_be_nonempty_and_unique(
        cls, keywords: list[str]
    ) -> list[str]:
        if any(not keyword for keyword in keywords):
            raise ValueError("keywords must not be empty")
        duplicates = _duplicate_values(keywords)
        if duplicates:
            raise ValueError(f"duplicate keywords: {', '.join(duplicates)}")
        return keywords

    @model_validator(mode="after")
    def claims_must_be_unique(self) -> "ResearchProblem":
        duplicates = _duplicate_values([claim.claim for claim in self.claims])
        if duplicates:
            raise ValueError(f"duplicate claims: {', '.join(duplicates)}")
        return self


class ExtractionResult(SchemaModel):
    problems: list[ResearchProblem] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def problems_must_be_unique(self) -> "ExtractionResult":
        duplicates = _duplicate_values(
            [problem.problem for problem in self.problems]
        )
        if duplicates:
            raise ValueError(f"duplicate problems: {', '.join(duplicates)}")
        return self


EXTRACTION_SYSTEM_PROMPT = """Extract a paper's problem, conceptual claims, and
implementation details. Return only information supported by the input text.

HIERARCHY

Problem
└── Claim
    └── Implementation detail

Most papers address one main problem. A problem may contain several claims,
and each claim may contain several implementation details.

DEFINITIONS

1. Problem — What limitation, failure, unmet need, or research gap motivates
the work?

Write one clear sentence. State the limitation itself, not the proposed
solution. Put domain and keywords on the problem, not on individual claims.

2. Claim — What new conceptual idea does the paper assert will address the
problem?

A claim is an independent, meaningful proposition. It must say more than
"we propose a model" or "performance improves." Keep concrete architecture,
loss, algorithm, and training choices out of the claim.

Multiple claims under one problem are allowed when they are distinct
conceptual assertions. Do not create separate claims for components, steps,
experiments, metrics, or implementation details.

3. Implementation detail — How do the authors make this claim happen?

Include concrete mechanisms such as architecture choices, prediction heads,
layer placement, pooling operations, objectives, losses, optimization
procedures, measurements, interventions, etc.

Use the replacement test: if this concrete choice changed but the parent claim
could remain true, it is an implementation detail. Explain its functional role
in domain-neutral language so it can be searched for alternatives.

Do NOT classify any of these as implementation details:
- experimental results or observed effects;
- ablations, benchmarks, or validation procedures;
- benefits such as "low overhead" or "improves accuracy";
- background methods merely used as baselines;
- restatements or fragments of the parent claim.

OUTPUT

Return 1-2 problems; almost always return exactly one. Each problem has:
- problem: one clear sentence;
- domain: a specific field or subfield;
- keywords: 3-6 short search labels covering the problem and claims;
- claims: 1-4 conceptual claims.

Each claim has:
- claim: one conceptual assertion;
- functional_role: its domain-neutral purpose;
- implementation_details: 0-5 concrete ways the paper realizes it.

Each implementation detail has:
- detail: the concrete technical choice;
- functional_role: what that choice accomplishes in domain-neutral terms.

EXAMPLE 1 — SEPARATE LOCALIZATION FROM CONFIDENCE

Input idea:
Existing methods lose spatial precision. The paper proposes separating
localization from confidence, implements this with two prediction heads,
trains with MSE and L1 losses, and validates the design by ablation.

Correct output:
{
  "problems": [
    {
      "problem": "Existing prediction methods lose spatial precision by
        coupling localization with confidence estimation.",
      "domain": "computer vision / object detection",
      "keywords": ["spatial precision", "localization", "confidence",
        "decoupled prediction"],
      "claims": [
        {
          "claim": "Separating localization from confidence preserves spatial
            precision.",
          "functional_role": "Decouples estimating where something is from
            estimating how certain the system is.",
          "implementation_details": [
            {
              "detail": "Use separate prediction heads for localization and
                confidence.",
              "functional_role": "Lets two outputs learn specialized
                representations instead of sharing one output pathway."
            },
            {
              "detail": "Train the outputs with a combined MSE and L1
                objective.",
              "functional_role": "Balances strong error penalization with a
                robust absolute-deviation signal."
            }
          ]
        }
      ]
    }
  ]
}

"Validated by ablation" is intentionally absent. It is evidence for the claim,
not a way the claim is implemented.

EXAMPLE 2 — GEOMETRY CONDITIONING

{
  "problems": [
    {
      "problem": "Learned representations are sensitive to perturbations
        because they do not use available geometric context.",
      "domain": "machine learning",
      "keywords": ["geometry conditioning", "robust representations",
        "feature modulation"],
      "claims": [
        {
          "claim": "Conditioning a learned representation on geometry improves
            robustness.",
          "functional_role": "Uses structural context to make a learned system
            less sensitive to perturbations.",
          "implementation_details": [
            {
              "detail": "Insert FiLM conditioning layers after every encoder
                block.",
              "functional_role": "Injects contextual information throughout
                successive stages of representation learning."
            }
          ]
        }
      ]
    }
  ]
}

BAD claim: "We insert FiLM layers after every encoder block."
That is an implementation detail, not a conceptual assertion (claim).

EXAMPLE 3 — CBAM

{
  "problems": [
    {
      "problem": "Convolutional networks do not explicitly emphasize the most
        informative channels and spatial locations in intermediate feature
        maps.",
      "domain": "computer vision / CNN architecture",
      "keywords": ["CBAM", "channel attention", "spatial attention",
        "feature refinement"],
      "claims": [
        {
          "claim": "Sequential attention at channel and spatial granularities
            improves feature refinement.",
          "functional_role": "Adaptively emphasizes useful parts of an
            intermediate representation in two stages at different
            granularities.",
          "implementation_details": [
            {
              "detail": "Apply channel attention first, followed by spatial
                attention.",
              "functional_role": "Refines a representation along one dimension
                before refining it along another."
            },
            {
              "detail": "Compute channel attention by combining average and max
                pooling through a shared MLP.",
              "functional_role": "Captures both global statistics and salient
                outliers with shared processing."
            },
            {
              "detail": "Compute spatial attention with a 7x7 convolution over
                pooled channel features.",
              "functional_role": "Uses a broad local neighborhood to estimate
                which positions matter."
            }
          ]
        }
      ]
    }
  ]
}

The statement "the module attaches to existing architectures with negligible
added parameters" is not an implementation detail. It describes compatibility
and cost. Do not output properties like this as mechanisms.

EXAMPLE 4 — LOCAL COMPLEXITY

{
  "problems": [
    {
      "problem": "Loss-based measures do not fully explain when deep neural
        networks transition from interpolation to robust generalization.",
      "domain": "machine learning / deep learning theory",
      "keywords": ["local complexity", "linear regions", "expressivity",
        "training dynamics"],
      "claims": [
        {
          "claim": "Changes in local complexity reveal a phase transition
            associated with delayed generalization and robustness.",
          "functional_role": "Uses a geometric signal to monitor how a model's
            representational behavior changes during training.",
          "implementation_details": [
            {
              "detail": "Define local complexity as the density of spline
                partition regions in the network's input space.",
              "functional_role": "Provides a geometric proxy for model
                expressivity that can be tracked over training."
            }
          ]
        }
      ]
    }
  ]
}

Do not create implementation details for "the curve shows double descent" or
"regions migrate toward the decision boundary." Those are observations unless
the text presents a concrete procedure that causes them.

BAD EXAMPLE — TOO GENERIC

{
  "problem": "AI needs improvement.",
  "claim": "Deep learning solves the problem.",
  "implementation_detail": "Use a neural network."
}

This is invalid because the problem, claim, and detail are generic and do not
capture the paper's specific contribution. Extract fewer items rather than
padding. Never invent missing details."""


def extract_claims(
    paper_text: str,
    *,
    client: OpenAI | None = None,
    model: str = DEFAULT_EXTRACTION_MODEL,
) -> list[ResearchProblem]:
    if not paper_text.strip():
        raise ValueError("paper_text must not be empty")

    completion = (client or get_client()).beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": paper_text},
        ],
        response_format=ExtractionResult,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The extraction model returned no structured result")
    return result.problems


def read_pdf_text(path: str, max_chars: int = 60000) -> str:
    """Extract text from a PDF, truncated to keep the context window reasonable."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars]


def build_output_path(input_path: str | None) -> Path:
    if input_path:
        stem = Path(input_path).stem.lower().replace(" ", "_")
    else:
        stem = "test_paper"
    return Path(f"{stem}_claims.json")


def save_claims(problems: list[ResearchProblem], output_path: Path) -> None:
    payload = {"problems": [problem.model_dump() for problem in problems]}
    output_path.write_text(ExtractionResult.model_validate(payload).model_dump_json(indent=2) + "\n")


DEFAULT_TEST_PAPER = """
We finetune the velocity field of a flow-matching generative model for
molecular reaction prediction using Group Relative Policy Optimization (GRPO).
Rewards are computed using AIMNet2, a pretrained neural network potential,
which scores generated molecular structures based on energy and force
plausibility. This provides a physics-grounded reward signal without
requiring expensive DFT calculations at each training step. We also
investigate whether combining this RL objective with an auxiliary MSE loss
on the ODE-integrated endpoint improves convergence, finding that the two
objectives produce conflicting gradients in early training.
"""


if __name__ == "__main__":
    import sys

    pdf_path = None
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Reading {pdf_path}...")
        paper_text = read_pdf_text(pdf_path)
    else:
        paper_text = DEFAULT_TEST_PAPER

    problems = extract_claims(paper_text)
    output_path = build_output_path(pdf_path)
    save_claims(problems, output_path)
    for problem in problems:
        print(problem.model_dump_json(indent=2))
    print(f"\nSaved claims to {output_path}")
