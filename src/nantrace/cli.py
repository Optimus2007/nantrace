"""nantrace command line.

    nantrace run train.py --lr 3e-4        trace a script, stop at the first NaN, explain it
    nantrace run --brief train.py          short JSON summary, sized for coding agents
    nantrace run --json train.py           full JSON report
    nantrace fuzz checks.py                run the fuzz_* functions in a file (see below)
    nantrace mcp                           start the MCP server for Claude Code, Cursor and others
    nantrace install-skill                 add the agent skill to .claude/skills in this project

`run` checks every op, so it is much slower than normal training. Point it at a
short reproduction (a few steps), or use nantrace.Guard inside long runs.

A fuzz file defines functions whose names start with fuzz_. Each one returns the
function to test and a valid example input:

    def fuzz_my_loss():
        return my_loss, (torch.randn(8, 10), torch.randint(0, 10, (8,)))

Exit codes: 0 nothing found, 3 NaN found (or fuzz findings), 4 an inf that is
still alive at the end but no NaN, 2 usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import sys

from .tracer import NonFiniteError, Report, Tracer

EXIT_FOUND = 3  # NaN found, or fuzz findings
EXIT_INF = 4  # no NaN, but inf created by something other than a mask, still alive at the end


def trace_script(target: str, args: list[str], stop: str = "nan", module: bool = False) -> tuple[Report | None, int, int]:
    """Run a script under the tracer. Returns (report, exit code, ops traced)."""
    sys.argv = [target, *args]
    if not module:
        sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    tracer = Tracer(stop=stop)
    report = None
    error = None
    keep = None  # the script's globals, so its tensors stay alive for the final check
    try:
        with tracer:
            if module:
                keep = runpy.run_module(target, run_name="__main__", alter_sys=True)
            else:
                keep = runpy.run_path(target, run_name="__main__")
    except NonFiniteError as e:
        report = e.report
    except SystemExit:
        pass
    except BaseException as e:  # autograd may wrap our error
        error = e

    if report is None and tracer.first_nan is not None:
        report = tracer.report()
    code = EXIT_FOUND if report is not None else 0
    if report is None:
        # No NaN. An inf that survives in a live tensor (a loss, a gradient, an output) can still
        # break training. Temporary infs that PyTorch masks away inside its own formulas are fine.
        live = tracer.live_events()
        ev = next((e for e in tracer.events if e.is_origin and not e.intended and not e.aux and e in live), None)
        if ev is not None:
            report = tracer.report(ev, {"summary": f"no NaN, but {ev.op} created inf from finite values"})
            code = EXIT_INF
    del keep
    if report is None and error is not None:
        raise error
    return report, code, tracer.n_ops


def _emit(payload: dict, out: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if out:
        with open(out, "w") as f:
            f.write(text + "\n")
    else:
        print(text)


def _cmd_run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="nantrace run", description="Trace a script and explain the first NaN.")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.add_argument("--brief", action="store_true", help="print a short JSON summary (for agents)")
    p.add_argument("--out", help="write the JSON to this file instead of stdout")
    p.add_argument("--stop", choices=["nan", "any"], default="nan", help="stop at the first nan (default) or the first nan or inf")
    p.add_argument("-m", dest="module", action="store_true", help="run a module, like python -m")
    p.add_argument("target", help="script path, or module name with -m")
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the script")
    ns = p.parse_args(argv)

    as_json = ns.json or ns.brief
    if not ns.module and not os.path.isfile(ns.target):
        msg = f"script not found: {os.path.abspath(ns.target)}"
        _emit({"found": False, "error": msg, "exit_code": 2}, ns.out) if as_json else print(f"nantrace: {msg}", file=sys.stderr)
        return 2
    # Keep stdout clean for the JSON: the script's own prints go to stderr.
    redirect = contextlib.redirect_stdout(sys.stderr) if as_json and not ns.out else contextlib.nullcontext()
    try:
        with redirect:
            report, code, n_ops = trace_script(ns.target, ns.args, ns.stop, ns.module)
    except Exception as e:  # the script failed for a reason other than NaN
        if not as_json:
            raise
        import traceback

        tb = traceback.extract_tb(e.__traceback__)
        from .tracer import _is_user_file

        user = [f for f in tb if _is_user_file(f.filename)][-1:] or tb[-1:]
        where = f" at {user[0].filename}:{user[0].lineno}: {(user[0].line or '').strip()}" if user else ""
        _emit({"found": False, "error": f"the script raised {type(e).__name__}: {e}{where}", "exit_code": 1}, ns.out)
        return 1

    if as_json:
        if report is None:
            payload = {"found": False, "summary": f"no NaN or Inf found ({n_ops} ops traced)"}
        else:
            payload = report.brief() if ns.brief else report.to_dict()
        payload["exit_code"] = code
        _emit(payload, ns.out)
    elif report is not None:
        print(report.text(), file=sys.stderr)
    else:
        print(f"nantrace: no NaN found ({n_ops} ops traced)", file=sys.stderr)
    return code


def load_fuzz_targets(path: str) -> list[tuple[str, object, tuple, dict]]:
    """Collect fuzz_* functions from a file. Each returns (fn, args) or (fn, args, kwargs)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    ns = runpy.run_path(path, run_name="nantrace_fuzz")
    targets = []
    for name, obj in ns.items():
        if name.startswith("fuzz_") and callable(obj):
            spec = obj()
            fn, args = spec[0], tuple(spec[1])
            kwargs = dict(spec[2]) if len(spec) > 2 else {}
            targets.append((name, fn, args, kwargs))
    return targets


def fuzz_file(path: str) -> list[dict]:
    from .fuzz import fuzz

    results = []
    for name, fn, args, kwargs in load_fuzz_targets(path):
        findings = fuzz(fn, *args, **kwargs)
        results.append(
            {
                "target": name,
                "findings": [
                    {"when": f.mutation, "also_when": f.also, "where": f.where, **f.report.brief()} for f in findings
                ],
                "_text": [str(f) for f in findings],
            }
        )
    return results


def _cmd_fuzz(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="nantrace fuzz", description="Run the fuzz_* functions in a file.")
    p.add_argument("--json", action="store_true", help="print JSON (short summaries, for agents and CI)")
    p.add_argument("--out", help="write the JSON to this file instead of stdout")
    p.add_argument("path", help="python file with fuzz_* functions")
    ns = p.parse_args(argv)

    redirect = contextlib.redirect_stdout(sys.stderr) if ns.json and not ns.out else contextlib.nullcontext()
    with redirect:
        results = fuzz_file(ns.path)
    total = sum(len(r["findings"]) for r in results)
    if not results:
        print(f"nantrace: no fuzz_* functions found in {ns.path}", file=sys.stderr)
        return 2
    if ns.json:
        _emit({"targets": [{k: v for k, v in r.items() if k != "_text"} for r in results], "total_findings": total}, ns.out)
    else:
        for r in results:
            n = len(r["findings"])
            print(f"{r['target']}: {n} finding{'s' if n != 1 else ''}", file=sys.stderr)
            for t in r["_text"]:
                print("\n" + t + "\n", file=sys.stderr)
    return EXIT_FOUND if total else 0


def _cmd_install_skill(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="nantrace install-skill", description="Copy the nantrace agent skill into a skills folder.")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--user", action="store_true", help="install for your user (~/.claude/skills) instead of this project")
    where.add_argument("--dir", help="install into this skills folder")
    ns = p.parse_args(argv)

    from importlib.resources import files

    src = files("nantrace").joinpath("SKILL.md").read_text()
    base = ns.dir or (os.path.expanduser("~/.claude/skills") if ns.user else os.path.join(os.getcwd(), ".claude", "skills"))
    dest_dir = os.path.join(base, "nantrace")
    dest = os.path.join(dest_dir, "SKILL.md")
    if os.path.exists(dest):
        with open(dest) as f:
            if f.read() == src:
                print(f"nantrace: skill already up to date at {dest}")
                return 0
        print(f"nantrace: {dest} exists and differs; not overwriting. Remove it first to reinstall.", file=sys.stderr)
        return 2
    os.makedirs(dest_dir, exist_ok=True)
    with open(dest, "w") as f:
        f.write(src)
    print(f"nantrace: installed the skill at {dest}")
    return 0


def _cmd_mcp(argv: list[str]) -> int:
    from .mcp_server import serve

    serve()
    return 0


COMMANDS = {"run": _cmd_run, "fuzz": _cmd_fuzz, "mcp": _cmd_mcp, "install-skill": _cmd_install_skill}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = COMMANDS.get(argv[0])
    if cmd is None:
        print(f"nantrace: unknown command {argv[0]!r}; commands: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    return cmd(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
