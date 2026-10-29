"""Hugging Face Space entry point for the Legal Clause Simplifier demo (not pushed by this repo).

Backend, chosen by environment variables:

* ``LCS_API_URL`` set: call a running ``lcs serve`` at that URL;
* otherwise: load the model in-process from ``LCS_SERVE_CONFIG`` (default
  ``configs/serve.yaml``) or, if ``LCS_MODEL`` is set, from that path / Hub id.
"""

from __future__ import annotations

import os
from pathlib import Path

from legal_simplifier.config import ModelConfig, ServeConfig, load_config
from legal_simplifier.serving.demo import Backend, LocalBackend, RemoteBackend, build_demo


def backend_from_env(environ: dict[str, str] | None = None) -> Backend:
    env = dict(os.environ if environ is None else environ)
    if env.get("LCS_API_URL"):
        return RemoteBackend(env["LCS_API_URL"])
    config_path = Path(env.get("LCS_SERVE_CONFIG", "configs/serve.yaml"))
    config = load_config(config_path, ServeConfig) if config_path.exists() else ServeConfig()
    if env.get("LCS_MODEL"):
        config = config.model_copy(
            update={
                "model": ModelConfig(
                    name_or_path=env["LCS_MODEL"],
                    revision=env.get("LCS_MODEL_REVISION"),
                    dtype=config.model.dtype,
                )
            }
        )
    return LocalBackend(config)


if __name__ == "__main__":
    build_demo(backend_from_env()).launch()
