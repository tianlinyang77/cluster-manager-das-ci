# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Bounded subprocess and SSH execution; importing this module does no I/O."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
from typing import Mapping, Sequence

from lib.evidence import STError


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _seconds(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("timeout must be a positive number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("timeout must be a positive finite number")
    return seconds


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _argv(values: Sequence[str | os.PathLike]) -> list[str]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("command must be a nonempty argv sequence")
    result = [os.fspath(value) for value in values]
    if any(not isinstance(value, str) or "\0" in value for value in result):
        raise ValueError("command arguments must be strings without NUL")
    return result


_SENSITIVE = re.compile(r"password|passwd|secret|token|private.?key|credential|authorization|api.?key", re.I)


def _display(argv: Sequence[str]) -> str:
    """Keep credential values out of the default command log."""
    result, hide_next = [], False
    for argument in argv:
        if hide_next:
            result.append("<redacted>")
            hide_next = False
        elif "=" in argument and _SENSITIVE.search(argument.split("=", 1)[0]):
            result.append(argument.split("=", 1)[0] + "=<redacted>")
        elif argument.startswith("-") and _SENSITIVE.search(argument):
            result.append(argument)
            hide_next = True
        else:
            result.append(argument)
    return shlex.join(result).replace("\r", "\\r").replace("\n", "\\n")


class Runner:
    """Run argv without a shell and own every process group it creates."""

    def __init__(self, evidence, command_timeout: float):
        self.evidence = evidence
        self.command_timeout = _seconds(command_timeout)
        self.environment = None
        self._owned: dict[int, subprocess.Popen] = {}

    @staticmethod
    def _group_options() -> dict:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def run(
        self,
        argv,
        *,
        timeout=None,
        input_text=None,
        env: Mapping[str, str] | None = None,
        cwd=None,
        check=True,
        category="environment",
    ) -> CommandResult:
        arguments = _argv(argv)
        limit = _seconds(self.command_timeout if timeout is None else timeout)
        display = _display(arguments)
        self.evidence.log(f"command timeout={limit:g}s: {display}")
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self.environment if env is None else env,
                cwd=cwd,
                **self._group_options(),
            )
        except OSError as error:
            raise STError(category, "COMMAND_START_FAILED", f"{display}: {error}") from error
        self._owned[process.pid] = process
        timed_out = False
        try:
            try:
                stdout, stderr = process.communicate(input_text, timeout=limit)
            except subprocess.TimeoutExpired as error:
                timed_out = True
                self.stop(process, min(limit, 5.0))
                try:
                    stdout, stderr = process.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    stdout, stderr = _text(error.stdout), _text(error.stderr)
                stderr = _text(stderr) + f"\nST command timeout after {limit:g}s"
            result = CommandResult(124 if timed_out else int(process.returncode), _text(stdout), _text(stderr))
            self.evidence.log(f"command returncode={result.returncode}: {display}")
            if check and result.returncode:
                failure = STError(
                    category,
                    "COMMAND_TIMEOUT" if timed_out else "COMMAND_FAILED",
                    f"returncode={result.returncode}: {display}",
                )
                failure.result = result
                raise failure
            return result
        finally:
            self._owned.pop(process.pid, None)

    def start(self, argv, logfile, *, env=None, cwd=None) -> subprocess.Popen:
        arguments = _argv(argv)
        path = Path(logfile).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence.log(f"start log={path}: {_display(arguments)}")
        try:
            with path.open("ab") as output:
                process = subprocess.Popen(
                    arguments,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=self.environment if env is None else env,
                    cwd=cwd,
                    **self._group_options(),
                )
        except OSError as error:
            raise STError("environment", "COMMAND_START_FAILED", f"{_display(arguments)}: {error}") from error
        self._owned[process.pid] = process
        self.evidence.logs[path.stem] = str(path)
        return process

    def stop(self, process: subprocess.Popen, timeout: float) -> int:
        """Stop only a group created by this Runner, including local children."""
        limit = _seconds(timeout)
        if self._owned.get(process.pid) is not process:
            if process.poll() is not None:
                return int(process.returncode)
            raise STError("test-framework", "PROCESS_NOT_OWNED", f"refusing to stop PID {process.pid}")
        self.evidence.log(f"stop owned process group pid={process.pid}")
        try:
            if os.name == "nt":
                if process.poll() is None:
                    try:
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    except OSError:
                        # Without a console, terminate() would orphan children.
                        killed = subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=limit,
                            check=False,
                        )
                        if killed.returncode and process.poll() is None:
                            raise OSError(f"taskkill failed with {killed.returncode}")
                    try:
                        process.wait(timeout=limit)
                    except subprocess.TimeoutExpired:
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=limit,
                            check=False,
                        )
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=limit)
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    pass
                # The leader can exit while its SSH/clush children still run.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=limit)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise STError("test-framework", "PROCESS_STOP_FAILED", f"PID {process.pid}: {error}") from error
        finally:
            self._owned.pop(process.pid, None)
        return int(process.returncode)


class Remote:
    """Execute a supplied script using the site's existing OpenSSH identity."""

    def __init__(self, runner: Runner, port=22, connect_timeout=10, ssh_options=()):
        self.runner = runner
        if isinstance(port, bool) or not str(port).isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError("SSH port must be in 1..65535")
        self.port = int(port)
        self.connect_timeout = max(1, math.ceil(_seconds(connect_timeout)))
        if isinstance(ssh_options, (str, bytes)):
            raise ValueError("ssh_options must be an argv list")
        self.ssh_options = _argv(ssh_options) if ssh_options else []
        compact = re.sub(r"[\s\"']+", "", " ".join(self.ssh_options)).lower()
        if re.search(r"stricthostkeychecking=?(?:no|off|false|0)(?:$|-)", compact):
            raise ValueError("disabling SSH host-key verification is not supported")
        if re.search(r"(?:user|global)knownhostsfile=?(?:/dev/null|nul|none)(?:$|-)", compact):
            raise ValueError("discarding SSH known-hosts checks is not supported")

    def script(self, host, script, args=(), *, timeout=None, check=True) -> CommandResult:
        target = str(host)
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:@%\[\]-]*", target):
            raise ValueError("invalid SSH host/alias")
        if not isinstance(script, str) or "\0" in script:
            raise ValueError("remote script must be text without NUL")
        if isinstance(args, (str, bytes)):
            raise ValueError("remote arguments must be an argv list")
        remote_command = shlex.join(["bash", "-s", "--", *[str(item) for item in args]])
        argv = [
            "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.connect_timeout}",
            "-p", str(self.port), *self.ssh_options, target, remote_command,
        ]
        return self.runner.run(argv, timeout=timeout, input_text=script, check=check)
