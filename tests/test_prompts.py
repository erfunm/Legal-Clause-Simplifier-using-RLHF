from __future__ import annotations

import pytest

from legal_simplifier.prompts import (
    SYSTEM_PROMPT,
    build_completion,
    build_messages,
    normalise_clause,
)

from .conftest import TINY_QWEN35, TINY_QWEN35_REVISION


def test_build_messages_structure() -> None:
    messages = build_messages("The Licensee shall pay.", "Cap On Liability")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert (
        messages[1]["content"]
        == "Clause type: Cap On Liability\n\nClause:\nThe Licensee shall pay."
    )


def test_build_messages_without_category() -> None:
    assert build_messages("Pay now.")[1]["content"] == "Clause:\nPay now."


def test_clause_whitespace_from_pdf_extraction_is_collapsed() -> None:
    assert normalise_clause("  The Party\n shall\t\tpay.  ") == "The Party shall pay."
    assert build_messages("a\n\nb")[1]["content"].endswith("Clause:\na b")


def test_empty_clause_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_messages(" \n ")


def test_system_prompt_states_the_style_guide() -> None:
    for requirement in ("grade 8", "party", "date", "amount", "defined term", "Do not add facts"):
        assert requirement in SYSTEM_PROMPT


def test_build_completion_strips_reference() -> None:
    assert build_completion("  You must pay.\n") == [
        {"role": "assistant", "content": "You must pay."}
    ]


@pytest.mark.slow
def test_chat_template_renders_prompt_as_prefix_of_full_conversation() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_QWEN35, revision=TINY_QWEN35_REVISION)
    prompt = build_messages("The Licensee shall pay $5,000.", "Cap On Liability")
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        prompt + build_completion("The Licensee must pay $5,000."), tokenize=False
    )
    assert isinstance(prompt_text, str)
    assert isinstance(full_text, str)
    assert full_text.startswith(prompt_text)
    assert "The Licensee shall pay $5,000." in prompt_text
    assert full_text[len(prompt_text) :].startswith("The Licensee must pay $5,000.")
