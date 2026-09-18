"""Daytona sandbox backend implementation."""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import Callable
from typing import cast
from uuid import uuid4

import daytona
from daytona import FileDownloadRequest, FileUpload, SessionExecuteRequest
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

SyncPollingInterval = float | Callable[[float], float]
PollingStrategy = Callable[[float], float]


def _resolve_polling_strategy(polling_interval: SyncPollingInterval) -> PollingStrategy:
    """Normalize a polling interval into a callable of elapsed time."""
    if callable(polling_interval):
        return cast("PollingStrategy", polling_interval)

    def polling_strategy(_elapsed: float) -> float:
        return polling_interval

    return polling_strategy


def _warn_async_method_on_sync_backend(method_name: str) -> None:
    """Warn that an async method was called on the sync `DaytonaSandbox`."""
    msg = (
        f"{method_name}() was called on a DaytonaSandbox, which only offers "
        "async methods by wrapping the synchronous path with "
        "asyncio.to_thread. For native async execution, use "
        "AsyncDaytonaSandbox with an async Daytona sandbox "
        "(daytona.AsyncSandbox) instead."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)


class DaytonaSandbox(BaseSandbox):
    """Daytona sandbox implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox
    and only implements the execute() method using Daytona's API.
    """

    def __init__(
        self,
        *,
        sandbox: daytona.Sandbox,
        timeout: int = 30 * 60,
        sync_polling_interval: SyncPollingInterval = 0.1,
    ) -> None:
        """Create a backend wrapping an existing Daytona sandbox.

        Args:
            sandbox: Existing Daytona sandbox instance to wrap.
            timeout: Default command timeout in seconds used when `execute()` is
                called without an explicit `timeout`.
            sync_polling_interval: Delay in seconds between polling Daytona for
                command completion on the sync execution path, or a callable
                that receives elapsed execution time in seconds and returns the
                next polling delay. This will eventually only appear on the
                sync path once an optimized async implementation is available.
        """
        self._sandbox = sandbox
        self._default_timeout = timeout
        self._sync_polling_interval = _resolve_polling_strategy(sync_polling_interval)

    @property
    def id(self) -> str:
        """Return the Daytona sandbox id."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                Note that in Daytona's implementation, a timeout of 0 means
                "wait indefinitely".
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        return self._execute_via_session_logs(command, timeout=effective_timeout)

    def _execute_via_session_logs(
        self,
        command: str,
        *,
        timeout: int,
    ) -> ExecuteResponse:
        """Execute a command through a session and poll logs until completion."""
        session_id = str(uuid4())
        self._sandbox.process.create_session(session_id)
        try:
            started_at = time.monotonic()
            result = self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=command, run_async=True),
                timeout=timeout,
            )
            while True:
                if timeout != 0 and time.monotonic() - started_at >= timeout:
                    msg = f"Command timed out after {timeout} seconds"
                    return ExecuteResponse(
                        output=msg,
                        exit_code=124,
                        truncated=False,
                    )
                command_result = self._sandbox.process.get_session_command(
                    session_id,
                    result.cmd_id,
                )
                if command_result.exit_code is not None:
                    break
                elapsed = time.monotonic() - started_at
                time.sleep(self._sync_polling_interval(elapsed))
            logs = self._sandbox.process.get_session_command_logs(
                session_id,
                result.cmd_id,
            )
        finally:
            self._sandbox.process.delete_session(session_id)

        output = logs.stdout or ""

        if logs.stderr is not None and logs.stderr.strip():
            output += f"\n<stderr>{logs.stderr.strip()}</stderr>"

        return ExecuteResponse(
            output=output,
            exit_code=command_result.exit_code,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
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

        if not download_requests:
            return responses

        daytona_responses = self._sandbox.fs.download_files(download_requests)

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

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        upload_requests: list[FileUpload] = []
        responses: list[FileUploadResponse] = []

        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            upload_requests.append(FileUpload(source=content, destination=path))
            responses.append(FileUploadResponse(path=path, error=None))

        if upload_requests:
            self._sandbox.fs.upload_files(upload_requests)

        return responses

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # forwarded to the sync path, not an asyncio timeout
    ) -> ExecuteResponse:
        """Async version of `execute`, wrapping the synchronous path.

        Emits a `UserWarning` because `DaytonaSandbox` has no native async
        support; use `AsyncDaytonaSandbox` instead.
        """
        _warn_async_method_on_sync_backend("aexecute")
        return await super().aexecute(command, timeout=timeout)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of `download_files`, wrapping the synchronous path.

        Emits a `UserWarning` because `DaytonaSandbox` has no native async
        support; use `AsyncDaytonaSandbox` instead.
        """
        _warn_async_method_on_sync_backend("adownload_files")
        return await super().adownload_files(paths)

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Async version of `upload_files`, wrapping the synchronous path.

        Emits a `UserWarning` because `DaytonaSandbox` has no native async
        support; use `AsyncDaytonaSandbox` instead.
        """
        _warn_async_method_on_sync_backend("aupload_files")
        return await super().aupload_files(files)


class AsyncDaytonaSandbox(BaseSandbox):
    """Async Daytona sandbox implementation conforming to SandboxBackendProtocol.

    This implementation inherits all async file operation methods from
    `BaseSandbox` and implements `aexecute()` natively using Daytona's async
    API. The synchronous `execute()`, `upload_files()`, and `download_files()`
    methods are not supported because the wrapped Daytona sandbox is async-only;
    use their async counterparts instead.

    Example:
        ```python
        from daytona import AsyncDaytona
        from langchain_daytona import AsyncDaytonaSandbox

        daytona_client = AsyncDaytona()
        sandbox = await daytona_client.create()
        backend = AsyncDaytonaSandbox(sandbox=sandbox)
        ```
    """

    def __init__(
        self,
        *,
        sandbox: daytona.AsyncSandbox,
        timeout: int = 30 * 60,
        polling_interval: SyncPollingInterval = 0.1,
    ) -> None:
        """Create a backend wrapping an existing async Daytona sandbox.

        Args:
            sandbox: Existing async Daytona sandbox instance to wrap.
            timeout: Default command timeout in seconds used when `aexecute()` is
                called without an explicit `timeout`.
            polling_interval: Delay in seconds between polling Daytona for command
                completion on the async execution path, or a callable that
                receives elapsed execution time in seconds and returns the next
                polling delay.
        """
        self._sandbox = sandbox
        self._default_timeout = timeout
        self._polling_interval = _resolve_polling_strategy(polling_interval)

    @property
    def id(self) -> str:
        """Return the Daytona sandbox id."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Not supported; use `aexecute()` instead."""
        msg = (
            "AsyncDaytonaSandbox does not support synchronous execution; "
            "use `await aexecute()` instead, or use the synchronous "
            "DaytonaSandbox class"
        )
        raise NotImplementedError(msg)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Not supported; use `aupload_files()` instead."""
        msg = (
            "AsyncDaytonaSandbox does not support synchronous uploads; "
            "use `await aupload_files()` instead, or use the synchronous "
            "DaytonaSandbox class"
        )
        raise NotImplementedError(msg)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Not supported; use `adownload_files()` instead."""
        msg = (
            "AsyncDaytonaSandbox does not support synchronous downloads; "
            "use `await adownload_files()` instead, or use the synchronous "
            "DaytonaSandbox class"
        )
        raise NotImplementedError(msg)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # forwarded to the sandbox, not an asyncio timeout
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If None, uses the backend's default timeout.

                Note that in Daytona's implementation, a timeout of 0 means
                "wait indefinitely".
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        return await self._execute_via_session_logs(command, timeout=effective_timeout)

    async def _execute_via_session_logs(
        self,
        command: str,
        *,
        timeout: int,  # noqa: ASYNC109  # forwarded to the sandbox, not an asyncio timeout
    ) -> ExecuteResponse:
        """Execute a command through a session and poll logs until completion."""
        session_id = str(uuid4())
        await self._sandbox.process.create_session(session_id)
        try:
            started_at = time.monotonic()
            result = await self._sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=command, run_async=True),
                timeout=timeout,
            )
            while True:
                if timeout != 0 and time.monotonic() - started_at >= timeout:
                    msg = f"Command timed out after {timeout} seconds"
                    return ExecuteResponse(
                        output=msg,
                        exit_code=124,
                        truncated=False,
                    )
                command_result = await self._sandbox.process.get_session_command(
                    session_id,
                    result.cmd_id,
                )
                if command_result.exit_code is not None:
                    break
                elapsed = time.monotonic() - started_at
                await asyncio.sleep(self._polling_interval(elapsed))
            logs = await self._sandbox.process.get_session_command_logs(
                session_id,
                result.cmd_id,
            )
        finally:
            await self._sandbox.process.delete_session(session_id)

        output = logs.stdout or ""

        if logs.stderr is not None and logs.stderr.strip():
            output += f"\n<stderr>{logs.stderr.strip()}</stderr>"

        return ExecuteResponse(
            output=output,
            exit_code=command_result.exit_code,
            truncated=False,
        )

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox."""
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

        if not download_requests:
            return responses

        daytona_responses = await self._sandbox.fs.download_files(download_requests)

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

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        upload_requests: list[FileUpload] = []
        responses: list[FileUploadResponse] = []

        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            upload_requests.append(FileUpload(source=content, destination=path))
            responses.append(FileUploadResponse(path=path, error=None))

        if upload_requests:
            await self._sandbox.fs.upload_files(upload_requests)

        return responses
