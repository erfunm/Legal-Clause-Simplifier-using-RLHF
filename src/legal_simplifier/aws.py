"""Optional S3 helpers (``pip install legal-simplifier[aws]``).

The core pipeline only reads and writes local paths. These helpers exist for
anyone who wants to stage data or checkpoints on S3 or SageMaker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""

    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an S3 URI, received: {uri}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI (missing bucket): {uri}")
    return bucket, key


def _client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "S3 support needs the 'aws' extra: pip install 'legal-simplifier[aws]'"
        ) from exc
    return boto3.client("s3")


def upload_file(local_path: Path | str, s3_uri: str) -> None:
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")
    bucket, key = parse_s3_uri(s3_uri)
    LOGGER.info("Uploading %s to %s", local_path, s3_uri)
    _client().upload_file(str(local_path), bucket, key)


def upload_directory(local_dir: Path | str, s3_uri: str) -> None:
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Cannot upload from missing directory: {local_dir}")
    client = _client()
    bucket, prefix = parse_s3_uri(s3_uri)
    prefix = prefix.rstrip("/")
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            relative = path.relative_to(local_dir).as_posix()
            client.upload_file(str(path), bucket, f"{prefix}/{relative}" if prefix else relative)


def download_file(s3_uri: str, local_path: Path | str) -> Path:
    bucket, key = parse_s3_uri(s3_uri)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(local_path))
    return local_path
