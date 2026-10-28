from __future__ import annotations

import pytest

from legal_simplifier.data.entities import (
    check_entity_presence,
    dedupe_entities,
    entity_recall,
    extract_entities,
    normalise_for_match,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("effective as of January 15, 2020 between", "January 15, 2020"),
        ("signed on the 3rd day of March, 2019.", "3rd day of March, 2019"),
        ("dated 15 June 2021 and", "15 June 2021"),
        ("expires on 12/31/2024.", "12/31/2024"),
        ("from Sept. 2022 onwards", "Sept. 2022"),
    ],
)
def test_extracts_dates(text: str, expected: str) -> None:
    assert expected in extract_entities(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a fee of $250,000 payable", "$250,000"),
        ("capped at USD 1,000,000.", "USD 1,000,000"),
        ("up to US$ 5 million in total", "US$ 5 million"),
        ("a sum of 10,000 dollars", "10,000 dollars"),
        ("liability of €2,500.50 maximum", "€2,500.50"),
    ],
)
def test_extracts_money(text: str, expected: str) -> None:
    assert expected in extract_entities(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a discount of 5% applies", "5%"),
        ("interest at 1.5 % per month", "1.5 %"),
        ("more than fifty percent (50 percent) of shares", "50 percent"),
    ],
)
def test_extracts_percentages(text: str, expected: str) -> None:
    assert expected in extract_entities(text)


def test_extracts_durations_with_number_words() -> None:
    entities = extract_entities("upon thirty (30) days written notice or within 5 business days")
    assert "thirty (30) days" in entities
    assert "5 business days" in entities


def test_extracts_quoted_defined_terms() -> None:
    entities = extract_entities('Globex Corporation (the "Supplier") and Hooli (“Customer”) agree.')
    assert "Supplier" in entities
    assert "Customer" in entities


def test_extracts_mid_sentence_capitalised_defined_terms() -> None:
    entities = extract_entities(
        "The Seller shall not compete within the Territory after the Effective Date."
    )
    assert {"Seller", "Territory", "Effective Date"} <= set(entities)
    assert "The Seller" not in entities


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Acme Widgets, Inc. shall pay", "Acme Widgets, Inc."),
        ("between Initech LLC and", "Initech LLC"),
        ("If Initech LLC is acquired", "Initech LLC"),
        ("The licensor is Hooli Ltd. of London", "Hooli Ltd."),
        ("Globex Corporation agrees", "Globex Corporation"),
    ],
)
def test_extracts_party_names(text: str, expected: str) -> None:
    entities = extract_entities(text)
    assert expected in entities
    assert not any(e.startswith(("If ", "The ")) for e in entities)


def test_ignores_shouted_boilerplate_and_generic_terms() -> None:
    entities = extract_entities("IN NO EVENT SHALL EITHER PARTY BE LIABLE under this Agreement.")
    assert entities == []


def test_drops_entities_subsumed_by_longer_ones() -> None:
    entities = extract_entities(
        "Payment is due on January 1, 2020 to Acme Widgets, Inc. by Acme Widgets."
    )
    assert "January" not in entities
    assert "Acme Widgets" not in entities
    assert "Acme Widgets, Inc." in entities


def test_returns_entities_in_order_without_duplicates() -> None:
    entities = extract_entities("Pay $100 on May 1, 2020. Again pay $100.")
    assert entities == ["$100", "May 1, 2020"]


def test_dedupe_uses_normalised_form() -> None:
    assert dedupe_entities(["USD 1,000", "$1000", " ", "Acme"]) == ["USD 1,000", "Acme"]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("USD 1,000,000", "$1000000"),
        ("thirty (30) days", "30 days"),
        ("5 percent", "5%"),
        ("January 1st, 2020", "january 1 2020"),
        ("“Supplier”", '"supplier"'),
    ],
)
def test_normalisation_equates_formatting_variants(left: str, right: str) -> None:
    assert normalise_for_match(left) == normalise_for_match(right)


def test_presence_tolerates_reformatting() -> None:
    entities = ["USD 1,000,000", "thirty (30) days", "Acme Widgets, Inc."]
    output = "Acme Widgets Inc. must pay $1,000,000 within 30 days."
    assert check_entity_presence(entities, output) == dict.fromkeys(entities, True)


def test_presence_requires_whole_token_match() -> None:
    presence = check_entity_presence(["30 days", "$100"], "You have 130 days to pay $1000.")
    assert presence == {"30 days": False, "$100": False}


def test_entity_recall() -> None:
    assert entity_recall(["$100", "Acme"], "Acme pays.") == 0.5
    assert entity_recall([], "anything") == 1.0
