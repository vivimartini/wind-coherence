"""Idempotent object writes for the live archive.

A second ingest of the same cycle replaces nothing and creates no duplicate.
Local disk and S3 share the interface, so the Fargate task and a laptop run
the same code. S3 writes land on a temporary key and are copied into place.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ObjectStore(Protocol):
    def put_bytes(self, key: str, data: bytes) -> str:
        """Write ``data`` at ``key``. Return ``written`` or ``unchanged``."""

    def exists(self, key: str) -> bool:
        """Whether an object is stored at ``key``."""


def _is_missing(exc: Exception) -> bool:
    name, text = type(exc).__name__, str(exc)
    return name in {"KeyError", "NoSuchKey"} or any(s in text for s in ("404", "Not Found", "NoSuchKey"))


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        dest = (self.root / key).resolve()
        root = self.root.resolve()
        if root not in dest.parents and dest != root:
            raise ValueError(f"key escapes archive root: {key}")
        return dest

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def put_bytes(self, key: str, data: bytes) -> str:
        dest = self.path(key)
        if dest.exists() and dest.read_bytes() == data:
            return "unchanged"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return "written"


class S3Store:
    def __init__(self, bucket: str, prefix: str = "", client=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.client = client

    def full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def _head(self, dest: str):
        try:
            return self.client.head_object(Bucket=self.bucket, Key=dest)
        except Exception as exc:
            if not _is_missing(exc):
                raise
            return None

    def exists(self, key: str) -> bool:
        return self._head(self.full_key(key)) is not None

    def put_bytes(self, key: str, data: bytes) -> str:
        dest = self.full_key(key)
        digest = sha256(data)
        head = self._head(dest)
        if head is not None and head.get("Metadata", {}).get("sha256") == digest:
            return "unchanged"
        partial = dest + ".partial"
        self.client.put_object(Bucket=self.bucket, Key=partial, Body=data, Metadata={"sha256": digest})
        self.client.copy_object(
            Bucket=self.bucket,
            Key=dest,
            CopySource={"Bucket": self.bucket, "Key": partial},
            Metadata={"sha256": digest},
            MetadataDirective="REPLACE",
        )
        self.client.delete_object(Bucket=self.bucket, Key=partial)
        return "written"


def open_store(url: str) -> ObjectStore:
    if url.startswith("s3://"):
        rest = url[len("s3://") :]
        bucket, _, prefix = rest.partition("/")
        return S3Store(bucket, prefix)
    return LocalStore(Path(url))
