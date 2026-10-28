"""W&B / Trackio setup for the Hugging Face ``Trainer``.

Credentials stay in the environment (``WANDB_API_KEY``, or ``wandb login``).
Trackio logs to a local store by default. Pointing it at a Hugging Face Space,
a Trackio server or a bucket publishes the dashboard, so that only happens
when the config names the Space explicitly: an ambient ``TRACKIO_SPACE_ID``
(or server/bucket variable) is refused rather than silently used.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from typing import Any

from legal_simplifier.config import TrackingConfig

TRACKIO_REMOTE_ENV = ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_REGISTRY_BUCKET_ID")


class TrackingPublishRefused(RuntimeError):
    pass


def configure_tracking(
    config: TrackingConfig, environ: MutableMapping[str, str] | None = None
) -> dict[str, Any]:
    """Set the tracker's environment and return the matching ``TrainingArguments`` kwargs."""

    env = os.environ if environ is None else environ
    if config.backend == "none":
        return {"report_to": ["none"]}

    if config.backend == "wandb":
        env["WANDB_PROJECT"] = config.project
        env["WANDB_MODE"] = config.wandb_mode
        if config.wandb_entity:
            env["WANDB_ENTITY"] = config.wandb_entity
        return {"report_to": ["wandb"], "project": config.project}

    remote = [name for name in TRACKIO_REMOTE_ENV if env.get(name)]
    if remote and config.trackio_space_id is None:
        raise TrackingPublishRefused(
            f"{', '.join(remote)} is set, which would send the Trackio dashboard off this "
            "machine. Unset it, or set tracking.trackio_space_id in the config to publish "
            "on purpose."
        )
    if config.trackio_dir is not None:
        env["TRACKIO_DIR"] = str(config.trackio_dir)
    return {
        "report_to": ["trackio"],
        "project": config.project,
        "trackio_space_id": config.trackio_space_id,
    }
