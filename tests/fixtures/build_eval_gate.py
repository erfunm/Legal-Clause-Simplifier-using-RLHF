"""Regenerate the eval-gate fixture in ``tests/fixtures/eval_gate/``. MOCK DATA ONLY.

The clauses and their "outputs" are synthetic, written for the tests: they are
not CUAD text, not model outputs and not results. ``examples.jsonl`` holds
faithful plain-English rewrites; ``examples_worse.jsonl`` is made from them by
rule (conditional parts dropped, some prohibitions flipped to permissions, one
entity removed) so the gate's tests can show it fails on a regression.
``thresholds.json`` is set from the good fixture's own scores with a tolerance.

Run: ``uv run python tests/fixtures/build_eval_gate.py``
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from legal_simplifier.data.entities import extract_entities
from legal_simplifier.eval.gate import gate_metrics
from legal_simplifier.rewards.components import clause_parts

OUT = Path(__file__).with_name("eval_gate")

PAIRS = [
    (
        "Termination For Convenience",
        "The Licensee may terminate this Agreement on thirty (30) days' written notice, unless the Licensor has cured the breach within ten (10) days.",
        "The Licensee can end this Agreement with 30 days' written notice, but not if the Licensor fixes the breach within 10 days.",
    ),
    (
        "Cap On Liability",
        "In no event shall Acme Widgets, Inc.'s aggregate liability exceed $250,000, except for liability arising from gross negligence.",
        "Acme Widgets, Inc. must not pay more than $250,000 in total, except for liability caused by gross negligence.",
    ),
    (
        "Renewal Term",
        "This Agreement shall automatically renew for successive one (1) year terms unless either party gives notice of non-renewal at least sixty (60) days before the end of the then-current term.",
        "This Agreement renews every 1 year unless either party gives notice that it will not renew at least 60 days before the current term ends.",
    ),
    (
        "Non-Compete",
        "During the Term and for twelve (12) months thereafter, the Distributor shall not sell any product that competes with the Products.",
        "During the Term and for 12 months after it, the Distributor must not sell any product that competes with the Products.",
    ),
    (
        "Anti-Assignment",
        "Neither party may assign this Agreement without the prior written consent of the other party, provided that such consent shall not be unreasonably withheld.",
        "Neither party may assign this Agreement without the other party's written consent first, provided that the other party does not refuse without a good reason.",
    ),
    (
        "Exclusivity",
        'Globex Corporation (the "Supplier") shall be the exclusive supplier of the Products in the Territory commencing January 1, 2020.',
        'From January 1, 2020, Globex Corporation (the "Supplier") is the only supplier of the Products in the Territory.',
    ),
    (
        "Governing Law",
        "This Agreement shall be governed by the laws of the State of New York, without regard to its conflict of laws principles.",
        "The laws of the State of New York apply to this Agreement, whatever its conflict of laws rules say.",
    ),
    (
        "Liquidated Damages",
        "If the Supplier fails to deliver by the Delivery Date, the Supplier shall pay the Buyer 2% of the order value for each week of delay, up to a maximum of 10%.",
        "If the Supplier delivers after the Delivery Date, the Supplier must pay the Buyer 2% of the order value for each week of delay, up to 10% in total.",
    ),
    (
        "Change Of Control",
        "In the event of a change of control of Initech LLC, the Licensor may terminate this Agreement upon ninety (90) days' notice.",
        "In the event of a change of control of Initech LLC, the Licensor can end this Agreement with 90 days' notice.",
    ),
    (
        "Ip Ownership Assignment",
        "All right, title and interest in any Deliverables shall vest in the Customer upon payment in full, subject to the Consultant's rights in its Background IP.",
        "The Customer owns all Deliverables once it has paid in full, subject to the Consultant's rights in its Background IP.",
    ),
    (
        "Post-Termination Services",
        "Following termination, the Supplier shall provide transition services for up to six (6) months at the rates then in effect.",
        "After termination, the Supplier must provide transition services for up to 6 months at the rates in force at that time.",
    ),
    (
        "Audit Rights",
        "The Licensor may audit the Licensee's records once per calendar year on fifteen (15) business days' notice, provided that the audit does not disrupt the Licensee's business.",
        "The Licensor can check the Licensee's records once a calendar year with 15 business days' notice, as long as the check does not disrupt the Licensee's business.",
    ),
    (
        "Insurance",
        "The Contractor shall maintain general liability insurance of not less than $1,000,000 per occurrence until the end of the Term.",
        "Until the Term ends, the Contractor must keep general liability insurance of at least $1,000,000 for each occurrence.",
    ),
    (
        "Minimum Commitment",
        "The Distributor shall purchase at least 10,000 units each calendar year, except in any year in which the Supplier fails to meet its delivery obligations.",
        "The Distributor must buy at least 10,000 units every calendar year, except in a year when the Supplier does not meet its delivery obligations.",
    ),
    (
        "Confidentiality",
        "Each party shall keep the Confidential Information confidential and shall not disclose it to any third party, except as required by law.",
        "Each party must keep the Confidential Information secret and must not share it with any third party, except when the law requires it.",
    ),
    (
        "Warranty Duration",
        "The Seller warrants that the Products will be free from defects for a period of twelve (12) months from delivery.",
        "The Seller promises that the Products will have no defects for 12 months from delivery.",
    ),
    (
        "Price Restrictions",
        "The Supplier shall not increase the prices set out in Schedule A more than once in any twelve (12) month period.",
        "The Supplier must not raise the prices in Schedule A more than once in any 12-month period.",
    ),
    (
        "Notice Period To Terminate Renewal",
        "Either party may terminate this Agreement for convenience upon one hundred twenty (120) days' prior written notice, provided that no termination shall take effect before December 31, 2025.",
        "Either party can end this Agreement for any reason with 120 days' written notice, but only if the end date is not before December 31, 2025.",
    ),
    (
        "Payment Terms",
        "The Customer shall pay each invoice within forty-five (45) days of receipt, unless the Customer disputes the invoice in good faith.",
        "The Customer must pay each invoice within 45 days of receiving it, unless the Customer disputes the invoice in good faith.",
    ),
    (
        "Third Party Beneficiary",
        "Nothing in this Agreement shall confer any rights on any person other than the parties to this Agreement.",
        "This Agreement gives no rights to anyone except the parties to this Agreement.",
    ),
]


def worse(clause: str, output: str, entities: list[str], index: int) -> str:
    """Rule-made regression: keep only the first conditional part, flip every third
    prohibition, and drop one entity from every second output."""

    parts = clause_parts(output)
    text = parts[0].rstrip(",;") + "." if len(parts) > 1 else output
    if index % 3 == 0:
        text = text.replace("must not", "may").replace("Neither party may", "Either party may")
    if index % 2 == 0 and entities:
        text = re.sub(re.escape(entities[-1]), "it", text, count=1, flags=re.IGNORECASE)
    return text


def main() -> None:
    OUT.mkdir(exist_ok=True)
    good, bad = [], []
    for i, (category, clause, output) in enumerate(PAIRS):
        entities = extract_entities(clause)
        row = {
            "id": f"gate-{i:02d}",
            "category": category,
            "clause": clause,
            "entities": entities,
            "reference": output,
            "reference_source": "mock",
            "output": output,
            "note": "MOCK: synthetic clause and hand-written output for the CI gate; not a result",
        }
        good.append(row)
        bad.append({**row, "output": worse(clause, output, entities, i)})
    for name, rows in (("examples.jsonl", good), ("examples_worse.jsonl", bad)):
        with (OUT / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    base = gate_metrics(good)
    thresholds = {
        "entity_recall": {"min": round(base["entity_recall"] - 0.05, 3)},
        "detector/condition_drop": {"max": round(base["detector/condition_drop"] + 0.05, 3)},
        "detector/modality_flip": {"max": round(base["detector/modality_flip"] + 0.05, 3)},
        **{k: {"min": round(v - 0.25, 2)} for k, v in base.items() if k.startswith("judge_proxy/")},
    }
    (OUT / "thresholds.json").write_text(
        json.dumps(
            {
                "_note": "MOCK fixture. Set from the good fixture's own scores (build_eval_gate.py) minus a tolerance.",
                "baseline": {k: round(v, 4) for k, v in base.items()},
                "thresholds": thresholds,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"good": base, "worse": gate_metrics(bad)}, indent=1))


if __name__ == "__main__":
    main()
