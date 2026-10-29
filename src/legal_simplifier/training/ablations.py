"""GRPO ablations: small YAML files of dotted overrides on ``configs/grpo.yaml``, one launcher.

An ablation file looks like::

    base: ../grpo.yaml                 # relative to this file
    name: ablation-entity-off
    description: GRPO without the entity reward in the advantage
    overrides:
      rewards.weights.entity_preservation: 0.0
      grpo.max_steps: 300

The resolved config gets ``experiment.name = name`` and
``grpo.output_dir = <output_root>/<name>``. Unknown keys fail validation, so a
typo cannot silently run the baseline. ``lcs ablate --dry-run`` prints every
run with its difference from the base config and trains nothing.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from legal_simplifier.config import GRPOConfigFile, StrictModel, load_yaml


class AblationSpec(StrictModel):
    name: str
    description: str = ""
    base: Path = Path("../grpo.yaml")
    output_root: Path = Path("checkpoints/ablations")
    overrides: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Ablation:
    path: Path
    spec: AblationSpec
    config: GRPOConfigFile
    changes: dict[str, tuple[Any, Any]]


def apply_overrides(data: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Set ``a.b.c: value`` keys in a copy of ``data`` (creating missing sections)."""

    result = copy.deepcopy(dict(data))
    for dotted, value in overrides.items():
        node = result
        *parents, leaf = dotted.split(".")
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"cannot set {dotted!r}: {part!r} is not a section")
            node = child
        node[leaf] = value
    return result


def _flatten(data: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(data, Mapping):
        for key, value in data.items():
            yield from _flatten(value, f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), data


def config_diff(base: GRPOConfigFile, other: GRPOConfigFile) -> dict[str, tuple[Any, Any]]:
    before = dict(_flatten(base.model_dump(mode="json")))
    after = dict(_flatten(other.model_dump(mode="json")))
    return {k: (before.get(k), v) for k, v in after.items() if before.get(k) != v}


def load_ablation(path: Path | str) -> Ablation:
    path = Path(path)
    spec = AblationSpec.model_validate(load_yaml(path))
    base_path = (path.parent / spec.base).resolve()
    base_data = load_yaml(base_path)
    resolved = apply_overrides(
        base_data,
        {
            **spec.overrides,
            "experiment.name": spec.name,
            "grpo.output_dir": str(spec.output_root / spec.name),
        },
    )
    config = GRPOConfigFile.model_validate(resolved)
    changes = config_diff(GRPOConfigFile.model_validate(base_data), config)
    return Ablation(path, spec, config, changes)


def discover(directory: Path | str, only: Sequence[str] = ()) -> list[Ablation]:
    """Every ``*.yaml`` in ``directory`` (sorted), optionally only the named ones."""

    ablations = [load_ablation(p) for p in sorted(Path(directory).glob("*.yaml"))]
    if only:
        unknown = set(only) - {a.spec.name for a in ablations}
        if unknown:
            raise ValueError(f"unknown ablation(s): {sorted(unknown)}")
        ablations = [a for a in ablations if a.spec.name in only]
    names = [a.spec.name for a in ablations]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate ablation names in {directory}: {names}")
    return ablations


def describe(ablation: Ablation) -> str:
    lines = [f"{ablation.spec.name}: {ablation.spec.description}".rstrip(": ")]
    for key, (old, new) in sorted(ablation.changes.items()):
        lines.append(f"    {key}: {old!r} -> {new!r}")
    return "\n".join(lines)


def run_ablations(ablations: Sequence[Ablation]) -> list[Path]:
    """Train each ablation in turn (one GPU). The Modal script runs them in parallel instead."""

    from legal_simplifier.training.grpo import train_grpo

    return [train_grpo(a.config) for a in ablations]
