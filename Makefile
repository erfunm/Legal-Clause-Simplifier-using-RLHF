# Common tasks. Everything here runs on CPU.
export HF_HUB_DISABLE_TELEMETRY := 1

SMOKE_CONFIG := configs/smoke/data.yaml

.PHONY: install lint format typecheck test test-all smoke clean

install:  ## Create .venv with every extra and the dev tools
	uv sync --all-extras

lint:  ## Ruff lint + format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:  ## Apply ruff fixes and formatting
	uv run ruff check --fix src tests scripts
	uv run ruff format src tests scripts

typecheck:
	uv run pyright

test:  ## Fast unit tests (no model downloads)
	uv run pytest -m "not slow"

test-all:  ## Unit tests plus tiny-model tests
	uv run pytest

smoke:  ## Data -> SFT -> GRPO -> DPO -> hacking scan on fixtures with tiny random models (outputs are not results)
	rm -rf data/smoke
	uv run lcs prepare-data --config $(SMOKE_CONFIG)
	uv run lcs generate-references --config $(SMOKE_CONFIG)
	uv run lcs build-golden --config $(SMOKE_CONFIG)
	uv run lcs dataset-card --config $(SMOKE_CONFIG)
	uv run lcs train-sft --config configs/smoke/sft.yaml
	uv run lcs train-grpo --config configs/smoke/grpo.yaml
	uv run lcs build-preferences --config configs/smoke/dpo.yaml
	uv run lcs train-dpo --config configs/smoke/dpo.yaml
	uv run lcs detect-hacking --outputs data/smoke/preferences/samples_train.jsonl
	uv run lcs ablate --dry-run

clean:
	rm -rf data/smoke checkpoints/smoke .pytest_cache .ruff_cache
