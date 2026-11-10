"""MCP server so coding agents can call nantrace as a tool.

Add it to Claude Code (use the Python of the environment that has your project's torch):

    claude mcp add nantrace -- /path/to/venv/bin/python -m nantrace mcp

Each tool call runs the script in a separate process with a timeout, so a crash
or a hang in user code cannot take the server down.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

INSTRUCTIONS = (
    "nantrace finds where NaN or Inf first appears in PyTorch code, explains why, and suggests a fix. "
    "Use trace_nan on a short script that reproduces the NaN. Use fuzz to test a loss, layer or function "
    "on edge case inputs before training. Fix the line in cause.location (or cause.inside_library for "
    "library code) instead of hiding the NaN with nan_to_num or clamping the loss."
)

FUZZ_FILE_HELP = """\
The file must define functions whose names start with fuzz_. Each returns the function (or nn.Module)
to test and a valid example input, as (fn, args) or (fn, args, kwargs):

    import torch
    from mymodel import my_loss

    def fuzz_my_loss():
        return my_loss, (torch.randn(8, 10), torch.randint(0, 10, (8,)))
"""


def _tail(text: str, n: int = 30) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


def _call(cmd_args: list[str], cwd: str | None, timeout: int, python: str | None) -> dict:
    fd, out_path = tempfile.mkstemp(prefix="nantrace_", suffix=".json")
    os.close(fd)
    cmd = [python or sys.executable, "-m", "nantrace", cmd_args[0], "--out", out_path, *cmd_args[1:]]
    try:
        proc = subprocess.run(cmd, cwd=cwd or None, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        try:
            os.remove(out_path)
        except OSError:
            pass
        return {
            "error": f"timed out after {timeout} s; make the reproduction shorter (fewer steps, smaller batch)",
            "output_tail": _tail((e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")),
        }
    try:
        with open(out_path) as f:
            result = json.load(f)
    except (OSError, ValueError):
        result = {"error": f"nantrace exited with code {proc.returncode} without a report"}
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
    result["exit_code"] = proc.returncode
    out = _tail(proc.stdout + proc.stderr)
    if out:
        result["output_tail"] = out
    return result


def build_server():
    try:  # mcp 2.x
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:  # mcp 1.x
        try:
            from mcp.server.fastmcp import FastMCP as Server
        except ImportError as e:
            raise SystemExit("nantrace mcp needs the MCP SDK: pip install 'nantrace[mcp]'") from e

    server = Server("nantrace", instructions=INSTRUCTIONS)

    @server.tool()
    def trace_nan(
        script: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        stop: str = "nan",
        timeout_seconds: int = 600,
        python: str | None = None,
    ) -> dict:
        """Run a Python script with every PyTorch op traced, and explain the first NaN.

        Returns found, summary, and cause: the op, the source line (location), the line inside
        a library when there is one (inside_library), the input ranges, why it happened and a
        suggested fix. chain shows how the value then spread. exit_code 3 means NaN, 4 means an
        Inf that is still alive at the end (for example an infinite loss) but no NaN.

        Tracing is slow, so point it at a short reproduction: a few training steps, a small batch.
        stop="any" also stops at the first Inf. python selects a different interpreter (for
        example the project's venv) if the server runs elsewhere.
        """
        return _call(["run", "--brief", "--stop", stop, script, *(args or [])], cwd, timeout_seconds, python)

    @server.tool(description="Test functions or modules on edge case inputs and report NaN or Inf bugs with root causes.\n\n" + FUZZ_FILE_HELP + (
        "\nEach target is tried with zeros, constant values, huge and tiny values, negative values, a batch of one, "
        "float16 and bfloat16, fully masked rows, and integer targets that are all one value (such as -100, the "
        "default ignore_index). Forward and backward are both checked. Findings with the same root cause are merged."
    ))
    def fuzz(path: str, cwd: str | None = None, timeout_seconds: int = 600, python: str | None = None) -> dict:
        return _call(["fuzz", "--json", path], cwd, timeout_seconds, python)

    return server


def serve() -> None:
    build_server().run("stdio")
