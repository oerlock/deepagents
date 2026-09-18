"""Shared helpers for the Daytona sandbox backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import daytona
from daytona import FileDownloadRequest, FileUpload
from deepagents.backends.protocol import (
    FileDownloadResponse,
    FileUploadResponse,
)

SyncPollingInterval = float | Callable[[float], float]
PollingStrategy = Callable[[float], float]


def resolve_polling_strategy(polling_interval: SyncPollingInterval) -> PollingStrategy:
    """Normalize a polling interval into a callable of elapsed time."""
    if callable(polling_interval):
        return cast("PollingStrategy", polling_interval)

    def polling_strategy(_elapsed: float) -> float:
        return polling_interval

    return polling_strategy


def build_download_requests(
    paths: list[str],
) -> tuple[list[FileDownloadRequest], list[FileDownloadResponse]]:
    """Split paths into Daytona download requests and placeholder responses.

    Non-absolute paths are rejected in place with `invalid_path` errors; the
    returned responses align by index with the input `paths`.
    """
    download_requests: list[FileDownloadRequest] = []
    responses: list[FileDownloadResponse] = []

    for path in paths:
        if not path.startswith("/"):
            responses.append(
                FileDownloadResponse(path=path, content=None, error="invalid_path")
            )
            continue
        download_requests.append(FileDownloadRequest(source=path))
        responses.append(FileDownloadResponse(path=path, content=None, error=None))

    return download_requests, responses


def map_download_responses(
    paths: list[str],
    daytona_responses: list[daytona.FileDownloadResponse],
    responses: list[FileDownloadResponse],
) -> list[FileDownloadResponse]:
    """Map Daytona download results back onto the placeholder responses."""
    mapped_responses: list[FileDownloadResponse] = []
    for resp in daytona_responses:
        content = resp.result
        if content is None:
            mapped_responses.append(
                FileDownloadResponse(
                    path=resp.source,
                    content=None,
                    error="file_not_found",
                )
            )
        else:
            mapped_responses.append(
                FileDownloadResponse(
                    path=resp.source,
                    content=content,  # ty: ignore[invalid-argument-type]  # Daytona SDK returns bytes for file content
                    error=None,
                )
            )

    mapped_iter = iter(mapped_responses)
    for i, path in enumerate(paths):
        if not path.startswith("/"):
            continue
        responses[i] = next(
            mapped_iter,
            FileDownloadResponse(path=path, content=None, error="file_not_found"),
        )

    return responses


def build_upload_requests(
    files: list[tuple[str, bytes]],
) -> tuple[list[FileUpload], list[FileUploadResponse]]:
    """Split files into Daytona upload requests and placeholder responses.

    Non-absolute paths are rejected in place with `invalid_path` errors; the
    returned responses align by index with the input `files`.
    """
    upload_requests: list[FileUpload] = []
    responses: list[FileUploadResponse] = []

    for path, content in files:
        if not path.startswith("/"):
            responses.append(FileUploadResponse(path=path, error="invalid_path"))
            continue
        upload_requests.append(FileUpload(source=content, destination=path))
        responses.append(FileUploadResponse(path=path, error=None))

    return upload_requests, responses
