"""Command line: run a script and stop at the first NaN with an explanation.

    nantrace run train.py --lr 3e-4
    nantrace run --json train.py         # machine readable, for agents and CI
    nantrace run --stop any train.py     # also stop at the first inf

Every op is checked, so this is much slower than normal training. Point it at a
short reproduction (a few steps), or use nantrace.Guard inside long runs.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys

from .tracer import NonFiniteError, Tracer

EXIT_FOUND = 3  # NaN found
EXIT_INF = 4  # no NaN, but inf created by something other than a mask


def _run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="nantrace run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--stop", choices=["nan", "any"], default="nan", help="stop at the first nan (default) or any nan/inf")
    p.add_argument("-m", dest="module", action="store_true", help="run a module, like python -m")
    p.add_argument("target", help="script path, or module name with -m")
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the script")
    ns = p.parse_args(argv)

    sys.argv = [ns.target, *ns.args]
    if not ns.module:
        sys.path.insert(0, os.path.dirname(os.path.abspath(ns.target)))
    tracer = Tracer(stop=ns.stop)
    report = None
    error = None
    keep = None  # the script's globals, so its tensors stay alive for the final check
    try:
        with tracer:
            if ns.module:
                keep = runpy.run_module(ns.target, run_name="__main__", alter_sys=True)
            else:
                keep = runpy.run_path(ns.target, run_name="__main__")
    except NonFiniteError as e:
        report = e.report
    except SystemExit:
        pass
    except BaseException as e:  # autograd may wrap our error
        error = e

    if report is None and tracer.first_nan is not None:
        report = tracer.report()
    inf_only = False
    if report is None:  # no NaN, but an inf that is not a mask can still break training
        # Only infs that survive in a live tensor (a loss, a gradient, an output). Temporary infs
        # that PyTorch masks away inside its own formulas are not a problem.
        live = tracer.live_events()
        ev = next((e for e in tracer.events if e.is_origin and not e.intended and not e.aux and e in live), None)
        if ev is not None:
            inf_only = True
            report = tracer.report(ev, {"summary": f"no NaN, but {ev.op} created inf from finite values"})
    if report is None and error is not None:
        raise error

    if ns.json:
        out = report.to_dict() if report else {"found": False, "ops_traced": tracer.n_ops}
        print(json.dumps(out, indent=2, default=str))
    elif report:
        print(report.text(), file=sys.stderr)
    else:
        print(f"nantrace: no NaN found ({tracer.n_ops} ops traced)", file=sys.stderr)
    del keep
    if inf_only:
        return EXIT_INF
    return EXIT_FOUND if report and report.found else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: nantrace run [--json] [--stop nan|any] [-m] script.py [args ...]")
        return 0
    if argv[0] == "run":
        return _run(argv[1:])
    print(f"nantrace: unknown command {argv[0]!r} (try: nantrace run script.py)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
