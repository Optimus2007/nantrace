"""Op level tracing of NaN and Inf values.

`Tracer` is a TorchDispatchMode. It sees every aten op in the forward and the
backward pass, checks outputs for NaN and Inf, and remembers which op first
produced each non-finite tensor. That lets it walk back from a bad value (a NaN
loss, an inf gradient) to the op where the problem started, instead of the
place where it was noticed.

Tracing checks every op, so it is slow. `Guard` uses it only to replay a step
that already went wrong.
"""

from __future__ import annotations

import os
import re
import sysconfig
import traceback
import warnings
import weakref
from dataclasses import dataclass, field

import torch
from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map

from ._stats import TensorStats, has_nonfinite, is_float, stats
from .diagnose import diagnose

_SKIP_DIRS = tuple(
    os.path.realpath(p) + os.sep
    for p in (
        os.path.dirname(__file__),
        os.path.dirname(torch.__file__),
        sysconfig.get_paths()["stdlib"],
        sysconfig.get_paths()["scripts"],  # console entry points such as bin/nantrace
    )
)
_EXP_OPS = {"exp", "exp2"}
# exp(x) rounds to 0 below about log(smallest subnormal)
_TINY = {torch.float16: -17.3, torch.bfloat16: -92.0, torch.float32: -103.3, torch.float64: -744.4}
_TB_RE = re.compile(r'File "(.+?)", line (\d+), in ([^\n]+)(?:\n[ \t]*([^\n]*))?')


class NonFiniteError(RuntimeError):
    """Raised when a NaN (or Inf) is found. `.report` holds the full Report."""

    def __init__(self, report: "Report"):
        super().__init__(report.text())
        self.report = report


@dataclass
class Frame:
    file: str
    line: int
    func: str
    code: str

    def short(self) -> str:
        path = self.file
        if "site-packages" + os.sep in path:
            return f"{path.split('site-packages' + os.sep, 1)[1]}:{self.line} in {self.func}"
        try:
            rel = os.path.relpath(path)
            if not rel.startswith(".."):
                path = rel
        except ValueError:
            pass
        return f"{path}:{self.line} in {self.func}"

    def to_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "func": self.func, "code": self.code}


def _is_user_file(path: str) -> bool:
    if path.startswith("<"):
        return False
    return not os.path.realpath(path).startswith(_SKIP_DIRS)


def _forward_frames() -> list[Frame]:
    frames = [
        Frame(f.filename, f.lineno or 0, f.name, (f.line or "").strip())
        for f in traceback.extract_stack()
        if _is_user_file(f.filename)
    ]
    return frames


def _backward_frames(node) -> list[Frame]:
    tb = node.metadata.get("traceback_") if node is not None else None
    if not tb:
        return []
    entries = tb if isinstance(tb, list) else re.split(r"(?m)^(?=\s*File \")", str(tb))
    frames = []
    for entry in entries:  # one frame per entry; frozen frames have no code line
        m = _TB_RE.search(entry)
        if m and _is_user_file(m.group(1)):
            frames.append(Frame(m.group(1), int(m.group(2)), m.group(3).strip(), (m.group(4) or "").strip()))
    return frames


@dataclass(eq=False)
class Event:
    index: int  # op counter when this happened
    phase: str  # "forward", "backward" or "input"
    op: str
    kind: str  # "nan", "inf" or "nan+inf"
    output: TensorStats | None
    inputs: list[TensorStats] = field(default_factory=list)
    location: Frame | None = None
    stack: list[Frame] = field(default_factory=list)
    module: str | None = None
    grad_fn: str | None = None
    cause: str = ""
    fix: str = ""
    parent: "Event | None" = None

    @property
    def is_origin(self) -> bool:
        return self.parent is None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "phase": self.phase,
            "op": self.op,
            "kind": self.kind,
            "output": self.output.to_dict() if self.output else None,
            "inputs": [s.to_dict() for s in self.inputs],
            "location": self.location.to_dict() if self.location else None,
            "stack": [f.to_dict() for f in self.stack],
            "module": self.module,
            "grad_fn": self.grad_fn,
            "cause": self.cause,
            "fix": self.fix,
        }


def _kind(ts: list[torch.Tensor]) -> str:
    nan = any(bool(torch.isnan(t).any()) for t in ts)
    inf = any(bool(torch.isinf(t).any()) for t in ts)
    return "nan+inf" if nan and inf else ("nan" if nan else "inf")


def _recompute_float64(func, args, kwargs):
    """Run the op again in float64 on CPU. Finite there means a range problem."""
    if func._schema.is_mutable or "dtype" in kwargs or "device" in kwargs:
        return None, None

    def up(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu()
            return t.double() if t.is_floating_point() else t
        return t

    try:
        res = func(*tree_map(up, args), **tree_map(up, kwargs))
    except Exception:
        return None, None
    res = [t for t in tree_leaves(res) if is_float(t)]
    if not res:
        return None, None
    finite = not any(has_nonfinite(t) for t in res)
    absmax = max((float(t.abs().max()) for t in res if t.numel()), default=None) if finite else None
    return finite, absmax


class Tracer(TorchDispatchMode):
    """Records where NaN and Inf values are created.

    stop: None to record everything, "nan" to raise NonFiniteError at the first
    NaN, "any" to raise at the first op that creates NaN or Inf from finite inputs.
    """

    def __init__(self, *models: torch.nn.Module, stop: str | None = None, max_events: int = 1000):
        super().__init__()
        self.stop = stop
        self.max_events = max_events
        self.events: list[Event] = []
        self.first_nan: Event | None = None
        self.n_ops = 0
        self._by_id: dict[int, tuple] = {}
        self._by_storage: dict[int, Event] = {}
        self._modules: list[str] = []
        self.underflows: list[dict] = []  # exp ops that rounded values to exactly 0
        self.names: dict[int, str] = {}
        for m in models:
            for name, mod in m.named_modules():
                self.names[id(mod)] = name or type(mod).__name__
            for name, p in m.named_parameters():
                self.names[id(p)] = name
            for name, b in m.named_buffers():
                self.names[id(b)] = name

    # context management --------------------------------------------------
    def __enter__(self):
        self._prev_anomaly = (torch.is_anomaly_enabled(), torch.is_anomaly_check_nan_enabled())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Records forward tracebacks on autograd nodes so backward ops can
            # be mapped back to the line that created them.
            torch.autograd.set_detect_anomaly(True, check_nan=False)
        # Anomaly mode warns when our own error passes through backward; we report it better.
        self._saved_filters = warnings.filters[:]
        warnings.filterwarnings("ignore", message="Error detected in", category=UserWarning)
        self._hooks = [
            register_module_forward_pre_hook(self._enter_module),
            register_module_forward_hook(self._exit_module),
        ]
        return super().__enter__()

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            for h in self._hooks:
                h.remove()
            warnings.filters[:] = self._saved_filters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.autograd.set_detect_anomaly(*self._prev_anomaly)

    def _enter_module(self, module, args):
        cls = type(module).__name__
        name = self.names.get(id(module))
        self._modules.append(f"{name} ({cls})" if name and name != cls else cls)

    def _exit_module(self, module, args, out):
        if self._modules:
            self._modules.pop()

    # provenance ------------------------------------------------------------
    @staticmethod
    def _storage_key(t: torch.Tensor) -> int | None:
        try:
            return t.untyped_storage().data_ptr() or None
        except Exception:
            return None

    def _lookup(self, t: torch.Tensor) -> Event | None:
        hit = self._by_id.get(id(t))
        if hit is not None and hit[0]() is t:
            return hit[1]
        key = self._storage_key(t)
        return self._by_storage.get(key) if key else None

    def _register(self, t: torch.Tensor, ev: Event) -> None:
        try:
            self._by_id[id(t)] = (weakref.ref(t), ev)
        except TypeError:
            pass
        key = self._storage_key(t)
        if key:
            self._by_storage[key] = ev
        if len(self._by_id) > 200_000:
            self._by_id = {k: v for k, v in self._by_id.items() if v[0]() is not None}

    def _external(self, t: torch.Tensor) -> Event:
        name = self.names.get(id(t))
        if name:
            cause = f"'{name}' was already non-finite before this code ran (often a previous optimizer step)"
        else:
            cause = (
                "a tensor entered the traced code already non-finite "
                "(input data, a parameter, or a value computed before tracing started)"
            )
        ev = Event(
            index=self.n_ops,
            phase="input",
            op=f"<input {name}>" if name else "<input>",
            kind=_kind([t]),
            output=stats(t),
            cause=cause,
            fix="check the data pipeline and earlier steps; nantrace.Guard catches this one step earlier",
        )
        self._add(ev)
        self._register(t, ev)
        return ev

    def _add(self, ev: Event) -> None:
        if len(self.events) < self.max_events:
            self.events.append(ev)
        if "nan" in ev.kind and self.first_nan is None:
            self.first_nan = ev

    # the hook ----------------------------------------------------------------
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.n_ops += 1
        in_t = [t for t in tree_leaves((args, kwargs)) if is_float(t)]
        mutable = func._schema.is_mutable
        if mutable:  # inputs change in place, so look at them first
            pre_bad = [has_nonfinite(t) for t in in_t]
            pre_stats = [None if b else stats(t) for t, b in zip(in_t, pre_bad)]
            pre_kind = [_kind([t]) if b else None for t, b in zip(in_t, pre_bad)]

        out = func(*args, **kwargs)

        outs = [t for t in tree_leaves(out) if is_float(t)]
        if func.overloadpacket.__name__.rstrip("_") in _EXP_OPS and outs:
            self._note_underflow(func, in_t, outs[0], pre_stats if mutable else None)
        bad_outs = [t for t in outs if has_nonfinite(t)]
        if not bad_outs:
            return out

        if not mutable:
            pre_bad = [has_nonfinite(t) for t in in_t]
            pre_stats = None
            pre_kind = [_kind([t]) if b else None for t, b in zip(in_t, pre_bad)]

        parent = None
        for t, bad in zip(in_t, pre_bad):
            if not bad:
                continue
            src = self._lookup(t) or self._external(t)
            if parent is None or ("nan" in src.kind and "nan" not in parent.kind):
                parent = src

        out_kind = _kind(bad_outs)
        input_has_nan = any(k and "nan" in k for k in pre_kind)
        if parent is not None and not ("nan" in out_kind and not input_has_nan):
            for t in bad_outs:  # plain propagation
                self._register(t, parent)
            return out

        node = torch._C._current_autograd_node()
        phase = "backward" if node is not None else "forward"
        frames = _backward_frames(node) if node is not None else _forward_frames()
        inputs = [s if s is not None else stats(t) for t, s in zip(in_t, pre_stats or [None] * len(in_t))]
        out_stats = stats(bad_outs[0])
        f64, f64_max = (None, None)
        if parent is None:
            f64, f64_max = _recompute_float64(func, args, kwargs)
        grad_fn = type(node).__name__ if node is not None else None
        op = str(func)
        cause, fix = diagnose(op, phase, inputs, out_stats, grad_fn, f64, f64_max)
        if parent is None and "0 / 0" in cause:
            u = next((u for u in reversed(self.underflows) if self.n_ops - u["index"] <= 200 and u["phase"] == phase), None)
            if u is not None:
                loc = u["location"].short() if u["location"] else "an earlier line"
                cause += (
                    f". Likely root cause: exp underflowed to exactly 0 at {loc} "
                    f"(smallest input {u['min']:.4g}; in {u['dtype']} exp(x) is 0 below about {u['limit']:.1f}), "
                    "so the weights summed to 0"
                )
                fix = "subtract the max before exp: exp(x - x.amax(dim, keepdim=True)), or use torch.softmax"
        ev = Event(
            index=self.n_ops,
            phase=phase,
            op=op,
            kind=out_kind,
            output=out_stats,
            inputs=inputs,
            location=frames[-1] if frames else None,
            stack=frames[-6:],
            module=self._modules[-1] if self._modules and phase == "forward" else None,
            grad_fn=grad_fn,
            cause=cause,
            fix=fix,
            parent=parent,
        )
        self._add(ev)
        for t in bad_outs:
            self._register(t, ev)

        if self.stop == "any" or (self.stop == "nan" and "nan" in out_kind):
            raise NonFiniteError(self.report(ev))
        return out

    def _note_underflow(self, func, in_t, out_t, pre_stats) -> None:
        if not in_t or out_t.numel() == 0 or out_t.dtype not in _TINY:
            return
        x = in_t[0]
        with torch.no_grad():
            if not bool((out_t == 0).any()):
                return
            lo = pre_stats[0].min if pre_stats and pre_stats[0] is not None else float(x.detach().min())
        node = torch._C._current_autograd_node()
        frames = _backward_frames(node) if node is not None else _forward_frames()
        self.underflows.append({
            "index": self.n_ops,
            "phase": "backward" if node is not None else "forward",
            "location": frames[-1] if frames else None,
            "min": lo,
            "dtype": str(out_t.dtype).replace("torch.", ""),
            "limit": _TINY[out_t.dtype],
        })
        del self.underflows[:-20]

    # reports -----------------------------------------------------------------
    def explain(self, t: torch.Tensor, context: dict | None = None) -> "Report":
        """Report for one tensor, for example a NaN loss or gradient."""
        ev = self._lookup(t) if has_nonfinite(t) else None
        return self.report(ev, context)

    def report(self, target: Event | None = None, context: dict | None = None) -> "Report":
        if target is None:
            target = self.first_nan or next((e for e in self.events if e.is_origin), None)
        chain = []
        e = target
        while e is not None:
            chain.append(e)
            e = e.parent
        chain.reverse()
        others = [e for e in self.events if e.is_origin and e not in chain]
        return Report(chain=chain, n_ops=self.n_ops, others=others[:20], other_origins=len(others), context=context or {})


@dataclass
class Report:
    chain: list[Event]
    n_ops: int
    others: list[Event] = field(default_factory=list)
    other_origins: int = 0
    context: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.chain)

    @property
    def origin(self) -> Event | None:
        return self.chain[0] if self.chain else None

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "origin": self.origin.to_dict() if self.origin else None,
            "chain": [e.to_dict() for e in self.chain],
            "ops_traced": self.n_ops,
            "other_origins": self.other_origins,
            "others": [e.to_dict() for e in self.others],
            "context": self.context,
        }

    def text(self) -> str:
        lines = []
        ctx = self.context
        if ctx.get("summary"):
            lines.append(f"nantrace: {ctx['summary']}")
        if not self.chain:
            lines.append(
                ctx.get("not_found")
                or "nantrace: no op created NaN or Inf from finite inputs in the traced code"
            )
            return "\n".join(lines)
        last = self.chain[-1]
        if not ctx.get("summary"):
            lines.append(f"nantrace: {last.kind} first appeared in the {last.phase} pass (op {last.index} of {self.n_ops})")
        for i, e in enumerate(self.chain, 1):
            label = "origin" if i == 1 else "then"
            out = f"{e.output.short()}" if e.output else ""
            lines.append("")
            lines.append(f"  {i}. {label:<6} {e.phase:<8} {e.op}  ->  {out}")
            if e.grad_fn:
                lines.append(f"     grad of  {e.grad_fn}")
            if e.location:
                tag = "forward line" if e.phase == "backward" else "at"
                lines.append(f"     {tag:<8} {e.location.short()}")
                if e.location.code:
                    lines.append(f"              {e.location.code}")
            if e.module:
                lines.append(f"     module   {e.module}")
            for j, s in enumerate(e.inputs[:4]):
                lines.append(f"     input {j}  {s.short()}")
            if e.cause:
                lines.append(f"     why      {e.cause}")
            if e.fix:
                lines.append(f"     fix      {e.fix}")
        if self.other_origins:
            lines.append("")
            lines.append(
                f"  ({self.other_origins} other ops also created inf or nan from finite inputs, "
                "for example -inf attention masks; use --json to see them)"
            )
        return "\n".join(lines)

    __str__ = text
