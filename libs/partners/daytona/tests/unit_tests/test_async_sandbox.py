from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_daytona.async_sandbox import AsyncDaytonaSandbox

COMMAND_TIMEOUT_EXIT_CODE = 124


def _make_sandbox(
    *,
    polling_interval: float = 0.1,
) -> tuple[AsyncDaytonaSandbox, MagicMock]:
    mock_sdk = MagicMock()
    mock_sdk.id = "sb-123"
    sb = AsyncDaytonaSandbox(
        sandbox=mock_sdk,
        polling_interval=polling_interval,
    )
    return sb, mock_sdk


def test_async_id_returns_sandbox_id() -> None:
    sb, mock_sdk = _make_sandbox()
    assert sb.id == "sb-123"
    assert mock_sdk.id == "sb-123"


def test_async_sync_methods_not_supported() -> None:
    sb, _ = _make_sandbox()
    with pytest.raises(NotImplementedError):
        sb.execute("echo hello")
    with pytest.raises(NotImplementedError):
        sb.upload_files([("/sandbox/a.txt", b"content")])
    with pytest.raises(NotImplementedError):
        sb.download_files(["/sandbox/a.txt"])


async def test_aexecute_returns_stdout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.create_session = AsyncMock()
    mock_sdk.process.execute_session_command = AsyncMock(
        return_value=SimpleNamespace(cmd_id="c1")
    )
    mock_sdk.process.get_session_command = AsyncMock(
        side_effect=[SimpleNamespace(exit_code=0)]
    )
    mock_sdk.process.get_session_command_logs = AsyncMock(
        return_value=SimpleNamespace(stdout="hello world", stderr="")
    )
    mock_sdk.process.delete_session = AsyncMock()

    result = await sb.aexecute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False
    assert mock_sdk.process.create_session.call_count == 1
    assert mock_sdk.process.delete_session.call_count == 1


async def test_aexecute_appends_stderr_block() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.create_session = AsyncMock()
    mock_sdk.process.execute_session_command = AsyncMock(
        return_value=SimpleNamespace(cmd_id="c1")
    )
    mock_sdk.process.get_session_command = AsyncMock(
        side_effect=[SimpleNamespace(exit_code=1)]
    )
    mock_sdk.process.get_session_command_logs = AsyncMock(
        return_value=SimpleNamespace(stdout="out", stderr="err happened")
    )
    mock_sdk.process.delete_session = AsyncMock()

    result = await sb.aexecute("false")

    assert result.exit_code == 1
    assert result.output == "out\n<stderr>err happened</stderr>"


async def test_aexecute_polls_until_exit() -> None:
    sb, mock_sdk = _make_sandbox(polling_interval=0.25)
    mock_sdk.process.create_session = AsyncMock()
    mock_sdk.process.execute_session_command = AsyncMock(
        return_value=SimpleNamespace(cmd_id="c1")
    )
    mock_sdk.process.get_session_command = AsyncMock(
        side_effect=[
            SimpleNamespace(exit_code=None),
            SimpleNamespace(exit_code=None),
            SimpleNamespace(exit_code=0),
        ]
    )
    mock_sdk.process.get_session_command_logs = AsyncMock(
        return_value=SimpleNamespace(stdout="done", stderr="")
    )
    mock_sdk.process.delete_session = AsyncMock()

    with (
        patch(
            "langchain_daytona.async_sandbox.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
        patch(
            "langchain_daytona.async_sandbox.time.monotonic",
            side_effect=[0.0, 0.0, 0.5, 1.0, 1.5, 2.0],
        ),
    ):
        result = await sb.aexecute("sleep 5")

    assert result.exit_code == 0
    assert result.output == "done"
    assert [call.args[0] for call in mock_sleep.call_args_list] == [0.25, 0.25]


async def test_aexecute_timeout() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.process.create_session = AsyncMock()
    mock_sdk.process.execute_session_command = AsyncMock(
        return_value=SimpleNamespace(cmd_id="c1")
    )
    mock_sdk.process.get_session_command = AsyncMock(
        return_value=SimpleNamespace(exit_code=None)
    )
    mock_sdk.process.delete_session = AsyncMock()

    with (
        patch("langchain_daytona.async_sandbox.asyncio.sleep", new_callable=AsyncMock),
        patch(
            "langchain_daytona.async_sandbox.time.monotonic",
            side_effect=[0.0, 0.0, 0.0, 11.0],
        ),
    ):
        result = await sb.aexecute("sleep 999", timeout=10)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert "timed out" in result.output


async def test_aupload_files_invalid_and_valid_paths() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.fs.upload_files = AsyncMock()

    responses = await sb.aupload_files(
        [
            ("relative/path.txt", b"bad"),
            ("/sandbox/good.txt", b"good"),
        ]
    )

    assert responses[0].error == "invalid_path"
    assert responses[1].error is None
    assert mock_sdk.fs.upload_files.call_count == 1
    requests = mock_sdk.fs.upload_files.call_args.args[0]
    assert len(requests) == 1
    assert requests[0].destination == "/sandbox/good.txt"


async def test_adownload_files_maps_results() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.fs.download_files = AsyncMock(
        return_value=[
            SimpleNamespace(source="/sandbox/exists.txt", result=b"content"),
            SimpleNamespace(source="/sandbox/missing.txt", result=None),
        ]
    )

    responses = await sb.adownload_files(
        [
            "relative/path.txt",
            "/sandbox/exists.txt",
            "/sandbox/missing.txt",
        ]
    )

    assert responses[0].error == "invalid_path"
    assert responses[1].content == b"content"
    assert responses[1].error is None
    assert responses[2].error == "file_not_found"


async def test_adownload_files_skips_sdk_call_when_all_invalid() -> None:
    sb, mock_sdk = _make_sandbox()
    mock_sdk.fs.download_files = AsyncMock()

    responses = await sb.adownload_files(["relative/path.txt"])

    assert responses[0].error == "invalid_path"
    assert mock_sdk.fs.download_files.call_count == 0
