"""Build ranked suggestion cards from kept judged candidates.

Cards are grouped under the source problem / claim / implementation detail
they attach to (from the judge primary_level + matched query indices).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from judge_candidates import kept_candidates


RELATION_LABELS = {
    "same_problem": "Same problem",
    "claim_support": "Supports a claim",
    "claim_extension": "Extends a claim",
    "claim_qualification": "Qualifies a claim",
    "claim_contradiction": "Challenges a claim",
    "implementation_alternative": "Implementation alternative",
}

RELATION_ORDER = [
    "same_problem",
    "claim_support",
    "claim_extension",
    "claim_qualification",
    "claim_contradiction",
    "implementation_alternative",
]

EMBEDXIV_INTRO = (
    "EmbedXiv is a literature-aware suggestion engine for research drafts that "
    "works by keeping you updated on novel contributions within your field. It checks "
    "your draft against recent CS papers published on arXiv and points to "
    "places where the work may need a stronger comparison, a clearer claim, or "
    "a different technical choice."
)

LEVEL_LABELS = {
    "problem": "Problem",
    "claim": "Claim",
    "implementation": "Implementation",
}

# Judge primary_level → matched_queries.level
LEVEL_TO_QUERY_LEVEL = {
    "problem": "problem",
    "claim": "claim",
    "implementation": "implementation",
}


def _truncate(text: str, limit: int = 280) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if isinstance(item, dict):
        return item
    raise TypeError(f"Expected mapping or pydantic model, got {type(item)!r}")


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def paper_date(candidate: dict[str, Any]) -> str | None:
    """Prefer arXiv datestamp (YYYY-MM-DD); fall back to S2 year."""
    datestamp = str(candidate.get("datestamp") or "").strip()
    if datestamp:
        return datestamp[:10]
    year = candidate.get("year")
    if year is None:
        return None
    text = str(year).strip()
    return text or None


def _sort_key(candidate: dict[str, Any]) -> tuple:
    distance = candidate.get("best_distance")
    try:
        distance_value = float(distance)
    except (TypeError, ValueError):
        distance_value = 1e9
    citations = candidate.get("citationCount")
    try:
        citation_value = -int(citations)
    except (TypeError, ValueError):
        citation_value = 0
    return (distance_value, citation_value, str(candidate.get("arxiv_id", "")))


def _match_distance(match: dict[str, Any]) -> float:
    try:
        return float(match.get("distance"))
    except (TypeError, ValueError):
        return 1e9


def _pick_match(
    candidate: dict[str, Any], *, primary_level: str
) -> dict[str, Any]:
    matches = [
        match
        for match in candidate.get("matched_queries") or []
        if isinstance(match, dict)
    ]
    query_level = LEVEL_TO_QUERY_LEVEL.get(primary_level, "problem")
    leveled = [match for match in matches if match.get("level") == query_level]
    if leveled:
        return min(leveled, key=_match_distance)
    hierarchy = [
        match
        for match in matches
        if match.get("level") not in (None, "recommendation")
    ]
    if hierarchy:
        return min(hierarchy, key=_match_distance)
    return {}


def _lookup_node(
    problems: list[dict[str, Any]],
    *,
    primary_level: str,
    problem_index: int,
    claim_index: int | None,
    detail_index: int | None,
) -> dict[str, Any]:
    """Resolve hierarchy text for the attachment node."""
    if not problems:
        return {
            "node_kind": primary_level,
            "node_path": "?",
            "node_label": LEVEL_LABELS.get(primary_level, primary_level),
            "node_text": "",
            "sort_key": (2, problem_index, claim_index or 0, detail_index or 0),
        }

    pi = max(0, min(problem_index, len(problems) - 1))
    problem = problems[pi]
    claims = list(problem.get("claims") or [])

    if primary_level == "problem" or not claims:
        return {
            "node_kind": "problem",
            "node_path": str(pi),
            "node_label": "Problem",
            "node_text": str(problem.get("problem") or ""),
            "sort_key": (0, pi, -1, -1),
            "problem_index": pi,
            "claim_index": None,
            "detail_index": None,
        }

    ci = 0 if claim_index is None else max(0, min(claim_index, len(claims) - 1))
    claim = _as_dict(claims[ci])
    details = list(claim.get("implementation_details") or [])

    if primary_level == "claim" or not details or detail_index is None:
        return {
            "node_kind": "claim",
            "node_path": f"{pi}.{ci}",
            "node_label": "Claim",
            "node_text": str(claim.get("claim") or ""),
            "sort_key": (1, pi, ci, -1),
            "problem_index": pi,
            "claim_index": ci,
            "detail_index": None,
        }

    di = max(0, min(detail_index, len(details) - 1))
    detail = _as_dict(details[di])
    return {
        "node_kind": "implementation",
        "node_path": f"{pi}.{ci}.{di}",
        "node_label": "Implementation",
        "node_text": str(detail.get("detail") or ""),
        "sort_key": (2, pi, ci, di),
        "problem_index": pi,
        "claim_index": ci,
        "detail_index": di,
    }


def resolve_attachment(
    candidate: dict[str, Any],
    problems: list[Any] | None,
) -> dict[str, Any]:
    """Attach a candidate to one source problem / claim / detail node."""
    normalized = [_as_dict(problem) for problem in (problems or [])]
    frozen = candidate.get("attachment")
    if isinstance(frozen, dict) and frozen.get("node_kind") is not None:
        return _lookup_node(
            normalized,
            primary_level=str(frozen["node_kind"]),
            problem_index=_safe_int(frozen.get("problem_index")) or 0,
            claim_index=_safe_int(frozen.get("claim_index")),
            detail_index=_safe_int(frozen.get("detail_index")),
        )

    judgment = candidate.get("judgment") or {}
    primary_level = judgment.get("primary_level") or "problem"
    if primary_level not in LEVEL_TO_QUERY_LEVEL:
        primary_level = "problem"

    match = _pick_match(candidate, primary_level=primary_level)
    problem_index = _safe_int(match.get("problem_index"))
    claim_index = _safe_int(match.get("claim_index"))
    detail_index = _safe_int(match.get("detail_index"))

    # Infer level from indices when primary_level is vague vs match shape.
    if primary_level == "implementation" and detail_index is None:
        # Prefer an implementation match with a detail index if present.
        for match in sorted(
            candidate.get("matched_queries") or [], key=_match_distance
        ):
            if match.get("level") != "implementation":
                continue
            detail_index = _safe_int(match.get("detail_index"))
            claim_index = _safe_int(match.get("claim_index"))
            problem_index = _safe_int(match.get("problem_index"))
            if detail_index is not None:
                break

    if problem_index is None:
        problem_index = 0

    return _lookup_node(
        normalized,
        primary_level=primary_level,
        problem_index=problem_index,
        claim_index=claim_index,
        detail_index=detail_index,
    )


def build_suggestion_card(
    candidate: dict[str, Any],
    *,
    problems: list[Any] | None = None,
) -> dict[str, Any]:
    judgment = candidate.get("judgment") or {}
    relation = judgment.get("relation") or "same_problem"
    level = judgment.get("primary_level") or "problem"
    arxiv_id = str(candidate.get("arxiv_id", "")).strip()
    url = candidate.get("url") or (
        f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    )
    attachment = resolve_attachment(candidate, problems)
    return {
        "arxiv_id": arxiv_id,
        "title": (candidate.get("title") or "").strip() or arxiv_id,
        "url": url,
        "date": paper_date(candidate),
        "relation": relation,
        "relation_label": RELATION_LABELS.get(relation, relation),
        "primary_level": level,
        "primary_level_label": LEVEL_LABELS.get(level, level),
        "why": (judgment.get("why") or "").strip(),
        "abstract": _truncate(str(candidate.get("abstract") or ""), 320),
        "best_distance": candidate.get("best_distance"),
        "citation_count": candidate.get("citationCount"),
        "retrieval_source": candidate.get("retrieval_source", "faiss"),
        "recommended_from": candidate.get("recommended_from"),
        "stage": judgment.get("stage"),
        "node_kind": attachment["node_kind"],
        "node_path": attachment["node_path"],
        "node_label": attachment["node_label"],
        "node_text": attachment["node_text"],
        "node_sort_key": attachment["sort_key"],
        "problem_index": attachment.get("problem_index"),
        "claim_index": attachment.get("claim_index"),
        "detail_index": attachment.get("detail_index"),
    }


def build_suggestion_cards(
    candidates: list[dict[str, Any]],
    *,
    problems: list[Any] | None = None,
    kept_only: bool = True,
) -> list[dict[str, Any]]:
    pool = kept_candidates(candidates) if kept_only else list(candidates)
    cards = [
        build_suggestion_card(candidate, problems=problems) for candidate in pool
    ]
    cards.sort(
        key=lambda card: (
            tuple(card.get("node_sort_key") or (9, 0, 0, 0)),
            RELATION_ORDER.index(card["relation"])
            if card["relation"] in RELATION_ORDER
            else len(RELATION_ORDER),
            _sort_key(
                {
                    "best_distance": card.get("best_distance"),
                    "citationCount": card.get("citation_count"),
                    "arxiv_id": card.get("arxiv_id"),
                }
            ),
        )
    )
    return cards


def group_cards(
    cards: list[dict[str, Any]],
    *,
    problems: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """Flat sibling boxes: Problem, Claim, Implementation (not nested).

    A parent box (problem/claim) is still emitted when it has no suggestion
    cards of its own but a descendant does, so the hierarchy stays readable.
    """
    normalized = [_as_dict(problem) for problem in (problems or [])]
    problem_cards: dict[int, list[dict[str, Any]]] = {}
    claim_cards: dict[tuple[int, int], list[dict[str, Any]]] = {}
    detail_cards: dict[tuple[int, int, int], list[dict[str, Any]]] = {}

    for card in cards:
        pi = _safe_int(card.get("problem_index"))
        if pi is None:
            pi = 0
        ci = _safe_int(card.get("claim_index"))
        di = _safe_int(card.get("detail_index"))
        kind = card.get("node_kind") or "problem"
        if kind == "implementation" and ci is not None and di is not None:
            detail_cards.setdefault((pi, ci, di), []).append(card)
        elif kind == "claim" and ci is not None:
            claim_cards.setdefault((pi, ci), []).append(card)
        else:
            problem_cards.setdefault(pi, []).append(card)

    if not normalized:
        paths = sorted(
            {
                (
                    _safe_int(card.get("problem_index")) or 0,
                    _safe_int(card.get("claim_index")),
                    _safe_int(card.get("detail_index")),
                    card.get("node_kind"),
                    card.get("node_text") or "",
                )
                for card in cards
            }
        )
        synthetic: list[dict[str, Any]] = []
        by_pi: dict[int, dict[str, Any]] = {}
        for pi, ci, di, kind, text in paths:
            problem = by_pi.get(pi)
            if problem is None:
                problem = {
                    "problem": text if kind == "problem" else f"Problem {pi}",
                    "claims": [],
                }
                by_pi[pi] = problem
                synthetic.append(problem)
            if kind in {"claim", "implementation"} and ci is not None:
                claims = problem["claims"]
                while len(claims) <= ci:
                    claims.append(
                        {
                            "claim": (
                                text
                                if kind == "claim"
                                else f"Claim {pi}.{len(claims)}"
                            ),
                            "implementation_details": [],
                        }
                    )
                if kind == "claim":
                    claims[ci]["claim"] = text or claims[ci]["claim"]
                if kind == "implementation" and di is not None:
                    details = claims[ci]["implementation_details"]
                    while len(details) <= di:
                        details.append(
                            {"detail": text if len(details) == di else ""}
                        )
                    details[di]["detail"] = text or details[di]["detail"]
        normalized = synthetic

    boxes: list[dict[str, Any]] = []
    for pi, problem in enumerate(normalized):
        claims = list(problem.get("claims") or [])
        problem_has_descendants = False
        claim_blocks: list[tuple[int, dict[str, Any], list, list]] = []
        for ci, raw_claim in enumerate(claims):
            claim = _as_dict(raw_claim)
            details = list(claim.get("implementation_details") or [])
            detail_entries = []
            for di, raw_detail in enumerate(details):
                detail = _as_dict(raw_detail)
                d_cards = detail_cards.get((pi, ci, di), [])
                if d_cards:
                    detail_entries.append((di, detail, d_cards))
            c_cards = claim_cards.get((pi, ci), [])
            if c_cards or detail_entries:
                problem_has_descendants = True
                claim_blocks.append((ci, claim, c_cards, detail_entries))

        p_cards = problem_cards.get(pi, [])
        if p_cards or problem_has_descendants:
            boxes.append(
                {
                    "kind": "problem",
                    "label": "Problem",
                    "text": str(problem.get("problem") or ""),
                    "cards": p_cards,
                }
            )
        for ci, claim, c_cards, detail_entries in claim_blocks:
            boxes.append(
                {
                    "kind": "claim",
                    "label": "Claim",
                    "text": str(claim.get("claim") or ""),
                    "cards": c_cards,
                }
            )
            for di, detail, d_cards in detail_entries:
                boxes.append(
                    {
                        "kind": "implementation",
                        "label": "Implementation",
                        "text": str(detail.get("detail") or ""),
                        "cards": d_cards,
                    }
                )
    return boxes


def _render_card_markdown(card: dict[str, Any]) -> list[str]:
    title = card["title"]
    url = card["url"]
    heading = f"[{title}]({url})" if url else title
    lines = [f"### {heading}", ""]
    if card.get("date"):
        lines.append(f"- **Date:** {card['date']}")
    lines.append(f"- **arXiv:** `{card['arxiv_id']}`")
    lines.append(f"- **Relation:** {card['relation_label']}")
    if card.get("why"):
        lines.append(f"- **Why:** {card['why']}")
    if card.get("recommended_from"):
        lines.append(f"- **Via S2 graph from:** `{card['recommended_from']}`")
    if card.get("abstract"):
        lines.append(f"- **Abstract:** {card['abstract']}")
    lines.append("")
    return lines


def render_markdown(
    cards: list[dict[str, Any]],
    *,
    source: str | None = None,
    problems: list[Any] | None = None,
) -> str:
    lines: list[str] = ["# EmbedXiv suggestions", ""]
    if source:
        lines.append(f"{len(cards)} kept paper(s). Source: `{source}`")
    else:
        lines.append(f"{len(cards)} kept paper(s).")
    lines.append("")
    for box in group_cards(cards, problems=problems):
        lines.append(f"## {box.get('label') or 'Node'}")
        lines.append("")
        if box.get("text"):
            lines.append(box["text"])
            lines.append("")
        for card in box.get("cards") or []:
            lines.extend(_render_card_markdown(card))
    return "\n".join(lines).rstrip() + "\n"


def _render_card_html(card: dict[str, Any]) -> str:
    title = html.escape(card["title"])
    url = html.escape(card["url"] or "#")
    why = html.escape(card.get("why") or "")
    abstract = html.escape(card.get("abstract") or "")
    arxiv_id = html.escape(card["arxiv_id"])
    relation = html.escape(card.get("relation_label") or "")
    date = html.escape(str(card["date"])) if card.get("date") else ""
    meta_bits = [f'<span class="pill">{relation}</span>']
    if card.get("recommended_from"):
        seed = html.escape(str(card["recommended_from"]))
        meta_bits.append(f'<span class="pill">via {seed}</span>')
    date_html = f'<time class="card-date">{date}</time>' if date else ""
    return f"""
<article class="card">
  {date_html}
  <h3><a href="{url}">{title}</a></h3>
  <div class="meta">{" ".join(meta_bits)} <code>{arxiv_id}</code></div>
  <p class="why">{why}</p>
  <p class="abstract">{abstract}</p>
</article>
""".strip()


def _render_grid(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return ""
    return (
        '<div class="grid">'
        + " ".join(_render_card_html(card) for card in cards)
        + "</div>"
    )


def render_html(
    cards: list[dict[str, Any]],
    *,
    source: str | None = None,
    problems: list[Any] | None = None,
) -> str:
    sections = []
    for box in group_cards(cards, problems=problems):
        kind = html.escape(str(box.get("kind") or "problem"))
        label = html.escape(str(box.get("label") or "Node"))
        text = html.escape(str(box.get("text") or ""))
        sections.append(
            f"""
<section class="box box-{kind}">
  <header class="box-header">
    <p class="box-kicker">{label}</p>
    <h2>{text}</h2>
  </header>
  {_render_grid(box.get("cards") or [])}
</section>
""".strip()
        )

    if source:
        lede = (
            f"{len(cards)} kept paper(s). "
            f"Source: <code>{html.escape(source)}</code>"
        )
    else:
        lede = f"{len(cards)} kept paper(s)."
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EmbedXiv suggestions</title>
  <style>
    :root {{
      --bg: #e8eef2;
      --ink: #14202b;
      --muted: #4d5d6a;
      --card: #f7fafc;
      --line: #c5d0d8;
      --accent: #0b5f7a;
      --problem: #edf3f7;
      --claim: #edf3f7;
      --implementation: #edf3f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, #d9e7f0 0%, transparent 42%),
        linear-gradient(180deg, #f4f7f9, var(--bg));
      line-height: 1.5;
    }}
    main {{
      width: min(1100px, calc(100% - 2rem));
      margin: 2rem auto 3rem;
    }}
    h1 {{
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      font-size: clamp(2rem, 4vw, 3rem);
      letter-spacing: -0.03em;
      margin: 0 0 0.35rem;
    }}
    .lede, .source {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .intro {{
      max-width: none;
      width: 100%;
      margin: 0.9rem 0 1.1rem;
      padding: 0.85rem 1.1rem;
      background: var(--card);
      border: 1px solid var(--line);
      border-left: 3px solid var(--accent);
      color: var(--ink);
      font-size: 1rem;
      line-height: 1.6;
    }}
    .box {{
      margin-top: 1.25rem;
      border: 1px solid var(--line);
      padding: 1rem 1rem 1.05rem;
    }}
    .box-problem {{ background: var(--problem); }}
    .box-claim {{ background: var(--claim); }}
    .box-implementation {{ background: var(--implementation); }}
    .box-header {{
      margin: 0 0 0.85rem;
    }}
    .box-kicker {{
      margin: 0 0 0.2rem;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .box-header h2 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      font-size: 1.2rem;
      line-height: 1.35;
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 0.85rem;
    }}
    .card {{
      position: relative;
      background: var(--card);
      border: 1px solid var(--line);
      padding: 1rem 1.05rem 1.1rem;
      padding-top: 1.35rem;
      min-height: 100%;
    }}
    .card-date {{
      position: absolute;
      top: 0.55rem;
      right: 0.7rem;
      font-size: 0.75rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .card h3 {{
      font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
      font-size: 1.05rem;
      line-height: 1.3;
      margin: 0 0 0.55rem;
      padding-right: 4.5rem;
    }}
    .card a {{
      color: var(--ink);
      text-decoration-color: #7aa0b5;
    }}
    .card a:hover {{ color: var(--accent); }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      align-items: center;
      margin-bottom: 0.7rem;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    .pill {{
      border: 1px solid var(--line);
      padding: 0.1rem 0.45rem;
      background: #eef4f7;
    }}
    .why {{
      margin: 0 0 0.55rem;
      font-size: 0.98rem;
    }}
    .abstract {{
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.8em;
    }}
  </style>
</head>
<body>
  <main>
    <h1>EmbedXiv suggestions</h1>
    <p class="intro">{html.escape(EMBEDXIV_INTRO)}</p>
    <p class="lede">{lede}</p>
    {" ".join(sections) if sections else "<p>No kept papers.</p>"}
  </main>
</body>
</html>
"""


def write_suggestion_outputs(
    candidates: list[dict[str, Any]],
    output_json: Path,
    *,
    source: str | None = None,
    problems: list[Any] | None = None,
    kept_only: bool = True,
) -> dict[str, Any]:
    cards = build_suggestion_cards(
        candidates, problems=problems, kept_only=kept_only
    )
    groups = group_cards(cards, problems=problems)
    cards_md = output_json.with_name(output_json.stem + "_cards.md")
    cards_html = output_json.with_name(output_json.stem + "_cards.html")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cards_md.write_text(render_markdown(cards, source=source, problems=problems))
    cards_html.write_text(render_html(cards, source=source, problems=problems))
    return {
        "cards": cards,
        "groups": groups,
        "markdown": cards_md,
        "html": cards_html,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build EmbedXiv suggestion cards from judged results JSON."
    )
    parser.add_argument("results_json", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Base path; writes *_cards.md/html beside it",
    )
    parser.add_argument(
        "--include-dropped",
        action="store_true",
        help="Include non-kept candidates (default: kept only)",
    )
    args = parser.parse_args()
    payload = json.loads(args.results_json.read_text())
    base = args.output or args.results_json
    written = write_suggestion_outputs(
        list(payload.get("candidates", [])),
        base,
        source=payload.get("source"),
        problems=list(payload.get("problems") or []),
        kept_only=not args.include_dropped,
    )
    print(
        f"Wrote {len(written['cards'])} cards → "
        f"{written['html']} (+ md)"
    )


if __name__ == "__main__":
    main()
