from __future__ import annotations

from pathlib import Path

import pytest

from legal_simplifier.data.schema import ClauseRecord, read_records

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent

TINY_QWEN35 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration-NoThink"
TINY_QWEN35_REVISION = "c21e64ab01a61fe5675253768d359262f6a19ab6"


@pytest.fixture
def fixture_records() -> list[ClauseRecord]:
    """Hand-written synthetic clauses with mock references (not CUAD text)."""

    return read_records(FIXTURES / "clauses.jsonl")


def make_record(
    id: str,
    contract_id: str,
    clause: str,
    *,
    reference: str = "ref",
    reference_source: str = "mock",
    category: str = "Governing Law",
) -> ClauseRecord:
    return ClauseRecord.model_validate(
        {
            "id": id,
            "contract_id": contract_id,
            "category": category,
            "clause": clause,
            "reference": reference,
            "reference_source": reference_source,
        }
    )
