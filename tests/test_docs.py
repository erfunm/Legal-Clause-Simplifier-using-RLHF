"""The README and docs: every relative link and #anchor resolves, and no stale text remains."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import REPO_ROOT

DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "RUNBOOK.md",
    REPO_ROOT / "docs" / "decisions.md",
    REPO_ROOT / "docs" / "CV_BULLETS.md",
    REPO_ROOT / "space" / "README.md",
    REPO_ROOT / "tests" / "fixtures" / "eval_gate" / "README.md",
]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)|!\[[^\]]*\]\(([^)\s]+)\)")


def github_slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""

    text = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    return {
        github_slug(line.lstrip("#"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("#") and not line.startswith("#!")
    }


def _links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # code blocks are not links
    return [a or b for a, b in LINK.findall(text)]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_relative_links_and_anchors_resolve(doc: Path) -> None:
    broken = []
    for link in _links(doc):
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target, _, anchor = link.partition("#")
        path = (doc.parent / target).resolve() if target else doc
        if not path.exists():
            broken.append(f"{link}: no such file")
        elif anchor and path.suffix == ".md" and anchor not in anchors(path):
            broken.append(f"{link}: no heading #{anchor}")
    assert not broken, broken


def test_every_adr_is_in_the_index_and_linked_from_it() -> None:
    text = (REPO_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (ADR-\d{3}: .+)$", text, re.M)
    index = text.split("## Index", 1)[1].split("\n---", 1)[0]
    assert len(headings) == len(re.findall(r"^\| \[\d{3}\]", index, re.M))
    for heading in headings:
        assert f"(#{github_slug(heading)})" in index


def test_no_leftover_future_tense() -> None:
    for doc in (REPO_ROOT / "README.md", REPO_ROOT / "docs" / "RUNBOOK.md"):
        text = doc.read_text(encoding="utf-8").lower()
        for phrase in ("coming in", "later phase", "to follow", "will be added"):
            assert phrase not in text, f"{doc.name}: {phrase!r}"


def test_readme_keeps_the_generated_results_markers() -> None:
    from legal_simplifier.eval.report import END, START

    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count(START) == 1 and text.count(END) == 1 and text.index(START) < text.index(END)
