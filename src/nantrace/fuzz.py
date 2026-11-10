"""Find NaN and Inf bugs before training does: try edge case inputs on a function or module.

    findings = nantrace.fuzz(loss_fn, logits, targets)
    for f in findings:
        print(f)

Starting from a valid example input, each mutation pushes one argument toward a
numerical boundary: zeros, constant values, huge or tiny magnitudes, negative
values, a batch of one, float16 and bfloat16, masks with a fully masked row, and
integer targets that are all the same value (such as an ignore_index). Each
mutated call runs forward and backward under the tracer. A finding is kept when
the output or a gradient is not finite, with the root cause of the first NaN or
Inf. Findings with the same origin line are merged.

This is meant for tests and CI: library maintainers can fuzz a loss or a layer
once and catch the edge cases users would otherwise report as issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from ._stats import has_nonfinite
from .tracer import Report, Tracer


@dataclass
class Mutation:
    name: str
    arg: int  # index into the flattened tensor arguments
    fn: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Finding:
    mutation: str
    where: str  # "output" or "gradient"
    report: Report
    also: list[str] = field(default_factory=list)  # other mutations with the same origin

    @property
    def key(self) -> tuple:
        o = self.report.cause_event
        loc = (o.inner or o.location) if o else None
        return (o.op if o else None, loc.file if loc else None, loc.line if loc else None)

    def __str__(self) -> str:
        head = f"nantrace.fuzz: {self.where} becomes not finite when {self.mutation}"
        if self.also:
            head += f" (also: {', '.join(self.also[:4])})"
        return head + "\n" + self.report.text()


def _float_mutations(i: int, t: torch.Tensor) -> list[Mutation]:
    m = [
        Mutation(f"arg {i} is all zeros", i, torch.zeros_like),
        Mutation(f"arg {i} has some zeros", i, lambda x: x.masked_fill(torch.arange(x.numel(), device=x.device).reshape(x.shape) % 2 == 0, 0) if x.numel() else x),
        Mutation(f"arg {i} is constant", i, lambda x: torch.full_like(x, float(x.flatten()[0]) if x.numel() else 1.0)),
        Mutation(f"arg {i} is scaled by 1e4", i, lambda x: x * 1e4),
        Mutation(f"arg {i} is scaled by 1e-30", i, lambda x: x * 1e-30),
        Mutation(f"arg {i} is negated", i, lambda x: -x),
        Mutation(f"arg {i} is negative", i, lambda x: -x.abs() - 1),
    ]
    if t.dtype == torch.float32:
        m += [
            Mutation(f"arg {i} is float16", i, lambda x: x.half()),
            Mutation(f"arg {i} is bfloat16", i, lambda x: x.bfloat16()),
            Mutation(f"arg {i} is float16 and scaled by 100", i, lambda x: (x * 100).half()),
            Mutation(
                f"arg {i} is float16 with values up to 1% of the float16 limit",
                i,
                lambda x: (x / x.abs().max().clamp_min(1e-30) * 655).half(),
            ),
            Mutation(
                f"arg {i} is float16 with values up to 25% of the float16 limit",
                i,
                lambda x: (x / x.abs().max().clamp_min(1e-30) * 16384).half(),
            ),
        ]
    return m


def _int_mutations(i: int, t: torch.Tensor) -> list[Mutation]:
    out = [Mutation(f"arg {i} (integer) has every entry equal to its first value", i, lambda x: torch.full_like(x, int(x.flatten()[0]) if x.numel() else 0))]
    for v in (-100, 0):  # -100 is the default ignore_index of PyTorch losses
        out.append(Mutation(f"arg {i} (integer) is all {v}", i, lambda x, v=v: torch.full_like(x, v)))
    return out


def _bool_mutations(i: int, t: torch.Tensor) -> list[Mutation]:
    def first_row_false(x):
        x = x.clone()
        if x.dim() >= 1 and x.shape[0]:
            x[0] = False
        return x

    def first_row_true(x):
        x = x.clone()
        if x.dim() >= 1 and x.shape[0]:
            x[0] = True
        return x

    return [
        Mutation(f"arg {i} (mask) is all False", i, torch.zeros_like),
        Mutation(f"arg {i} (mask) is all True", i, torch.ones_like),
        Mutation(f"arg {i} (mask) has a first row of all False", i, first_row_false),
        Mutation(f"arg {i} (mask) has a first row of all True", i, first_row_true),
    ]


def _batch_mutations(i: int, t: torch.Tensor) -> list[Mutation]:
    if t.dim() >= 1 and t.shape[0] > 1:
        return [Mutation(f"arg {i} has batch size 1", i, lambda x: x[:1])]
    return []


def default_mutations(tensors: list[torch.Tensor]) -> list[Mutation]:
    muts: list[Mutation] = []
    for i, t in enumerate(tensors):
        if t.is_floating_point():
            muts += _float_mutations(i, t)
        elif t.dtype == torch.bool:
            muts += _bool_mutations(i, t)
        elif not t.is_complex():
            muts += _int_mutations(i, t)
    return muts


def _float_leaves(x: Any) -> list[torch.Tensor]:
    leaves, _ = tree_flatten(x)
    return [t for t in leaves if isinstance(t, torch.Tensor) and t.is_floating_point()]


def _run(fn, args, kwargs, models) -> tuple[str | None, Report | None]:
    """Call fn once under the tracer. Returns (where, report) when the result has NaN or Inf."""
    leaves, spec = tree_flatten((args, kwargs))
    leaves = [
        t.detach().clone().requires_grad_(True) if isinstance(t, torch.Tensor) and t.is_floating_point() else t
        for t in leaves
    ]
    a, k = tree_unflatten(leaves, spec)
    params = [p for m in models for p in m.parameters()]
    for p in params:
        p.grad = None
    tracer = Tracer(*models)
    with tracer:
        out = fn(*a, **k)
        outs = _float_leaves(out)
        bad = next((t for t in outs if has_nonfinite(t)), None)
        if bad is not None:
            return "output", tracer.explain(bad)
        graded = [t for t in outs if t.requires_grad]
        if not graded:
            return None, None
        total = sum(t.float().sum() for t in graded)
        total.backward()
    grads = [t.grad for t in leaves if isinstance(t, torch.Tensor) and t.grad is not None]
    grads += [p.grad for p in params if p.grad is not None]
    bad = next((g for g in grads if has_nonfinite(g)), None)
    if bad is not None:
        return "gradient", tracer.explain(bad)
    return None, None


def fuzz(
    fn: Callable,
    *args,
    mutations: list[Mutation] | None = None,
    **kwargs,
) -> list[Finding]:
    """Try edge case versions of the example inputs and report NaN and Inf bugs.

    `fn` can be a function or an nn.Module. Gradients of float inputs and of the
    module parameters are checked too. Mutations that make `fn` raise (for
    example a shape error) are skipped: those are not numerical bugs.
    """
    models = [fn] if isinstance(fn, torch.nn.Module) else []
    leaves, spec = tree_flatten((args, kwargs))
    tensor_pos = [j for j, t in enumerate(leaves) if isinstance(t, torch.Tensor)]
    tensors = [leaves[j] for j in tensor_pos]

    where, base = _run(fn, args, kwargs, models)
    if where is not None:
        return [Finding("the example input itself is used", where, base)]

    findings: dict[tuple, Finding] = {}
    for mut in mutations if mutations is not None else default_mutations(tensors):
        new = list(leaves)
        try:
            mutated = mut.fn(tensors[mut.arg])
            if mutated.is_floating_point() and has_nonfinite(mutated):
                continue  # the mutation itself overflowed; not a bug in fn
            new[tensor_pos[mut.arg]] = mutated
            a, k = tree_unflatten(new, spec)
            where, rep = _run(fn, a, k, models)
        except Exception:
            continue
        if where is None or rep is None or not rep.found:
            continue
        f = Finding(mut.name, where, rep)
        if f.key in findings:
            findings[f.key].also.append(mut.name)
        else:
            findings[f.key] = f
    return list(findings.values())
