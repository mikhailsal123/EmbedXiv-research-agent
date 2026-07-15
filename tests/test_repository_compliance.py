from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_repository_has_license_and_env_template() -> None:
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / ".env.example").is_file()


def test_no_local_papers_or_generated_data_are_tracked() -> None:
    tracked = _tracked_files()
    forbidden_prefixes = ("papers/", "data/", "output/")
    offenders = [
        path
        for path in tracked
        if path.startswith(forbidden_prefixes) or path.endswith((".pdf", ".faiss", ".sqlite"))
    ]
    assert offenders == []


def test_runtime_requirements_are_exactly_pinned() -> None:
    requirement_lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert requirement_lines
    assert all("==" in line for line in requirement_lines)
