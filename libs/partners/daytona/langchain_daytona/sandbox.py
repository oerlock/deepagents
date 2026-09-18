"""Daytona sandbox backend implementation."""

from __future__ import annotations

import time
import warnings
from uuid import uuid4

import daytona
from daytona import SessionExecuteRequest
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from langchain_daytona._utils import (
    SyncPollingInterval,
    build_download_requests,
    build_upload_requests,
    map_download_responses,
    resolve_polling_strategy,
)


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

    The async methods (`aexecute()`, `adownload_files()`, `aupload_files()`)
    are inherited defaults that wrap the synchronous path with
    `asyncio.to_thread`; calling them emits a `UserWarning`. For native async
    support, use `AsyncDaytonaSandbox`.
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
        self._sync_polling_interval = resolve_polling_strategy(sync_polling_interval)

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
        download_requests, responses = build_download_requests(paths)

        if not download_requests:
            return responses

        daytona_responses = self._sandbox.fs.download_files(download_requests)

        return map_download_responses(paths, daytona_responses, responses)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox."""
        upload_requests, responses = build_upload_requests(files)

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
