"""Cheap per step NaN check that replays the bad step with full tracing.

Training runs at normal speed. After each backward pass the guard checks that
the loss and the gradient norm are finite (one fused reduction and one sync).
Only when that check fails does it restore the random number state and the
module buffers from the start of the step, run the same step again under
`Tracer`, and report the op where the NaN or Inf started.

The check runs before `optimizer.step()`, so the parameters are still the ones
that produced the bad step and nothing needs to be copied.
"""

from __future__ import annotations

import random
from typing import Any, Callable

import torch

from .tracer import NonFiniteError, Report, Tracer


def _rng_state() -> dict:
    s: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        s["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        try:
            s["mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    try:
        import numpy as np

        s["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return s


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    random.setstate(s["python"])
    if "cuda" in s:
        torch.cuda.set_rng_state_all(s["cuda"])
    if "mps" in s:
        torch.mps.set_rng_state(s["mps"])
    if "numpy" in s:
        import numpy as np

        np.random.set_state(s["numpy"])


class Guard:
    """Wrap each training step to catch NaN and Inf at their source.

    Example::

        guard = nantrace.Guard(model)
        for batch in loader:
            loss = guard.step(compute_loss, batch)   # forward + backward
            optimizer.step()
            optimizer.zero_grad()

    `compute_loss(*args, **kwargs)` must return a scalar loss. The guard calls
    `loss.backward()` itself.

    on_error: "raise" (default) raises NonFiniteError with the report.
              "skip" prints the report, clears the gradients so the next
              `optimizer.step()` leaves the parameters alone, and returns None.
    check_grads: also check gradients. Turn this off with a GradScaler, which
              expects inf gradients now and then and skips those steps itself.
    """

    def __init__(
        self,
        *models: torch.nn.Module,
        check_grads: bool = True,
        on_error: str = "raise",
        report: Callable[[Report], None] | None = None,
    ):
        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        self.models = models
        self.check_grads = check_grads
        self.on_error = on_error
        self._report_fn = report or (lambda r: print(r.text()))
        self.steps = 0
        self.reports: list[Report] = []

    def _params(self):
        for m in self.models:
            yield from m.named_parameters()

    def _buffers(self):
        for m in self.models:
            yield from m.named_buffers()

    def _check(self, loss: torch.Tensor) -> tuple[bool, list[str]]:
        """One fused reduction and one device sync for the loss and all gradients."""
        named = [(n, p.grad) for n, p in self._params() if p.grad is not None] if self.check_grads else []
        vals = [loss.detach().float().sum().reshape(1)]
        if named:
            vals.append(torch.stack([v.float() for v in torch._foreach_norm([g for _, g in named])]))
        allv = torch.cat(vals)
        if bool(torch.isfinite(allv.sum())):
            return False, []
        flags = torch.isfinite(allv).tolist()
        return not flags[0], [n for (n, _), ok in zip(named, flags[1:]) if not ok]

    def step(self, fn: Callable[..., torch.Tensor], *args, **kwargs) -> torch.Tensor | None:
        self.steps += 1
        rng = _rng_state()
        buffers = {n: b.detach().clone() for n, b in self._buffers()}

        loss = fn(*args, **kwargs)
        loss.backward()

        loss_bad, bad_grads = self._check(loss)
        if not loss_bad and not bad_grads:
            return loss.detach()

        report = self._replay(fn, args, kwargs, rng, buffers, loss, loss_bad, bad_grads)
        self.reports.append(report)
        if self.on_error == "raise":
            raise NonFiniteError(report)
        self._report_fn(report)
        return None

    def _restore(self, rng: dict, buffers: dict) -> None:
        _set_rng_state(rng)
        current = dict(self._buffers())
        with torch.no_grad():
            for n, b in buffers.items():
                current[n].copy_(b)

    def _replay(self, fn, args, kwargs, rng, buffers, loss, loss_bad, bad_grads) -> Report:
        what = []
        if loss_bad:
            what.append(f"loss is {loss.detach().float().reshape(-1)[0].item()}")
        if bad_grads:
            shown = ", ".join(bad_grads[:3]) + (" ..." if len(bad_grads) > 3 else "")
            what.append(f"{len(bad_grads)} parameter gradients are non-finite ({shown})")
        context = {"step": self.steps, "loss_finite": not loss_bad, "bad_grads": bad_grads}
        context["summary"] = f"step {self.steps}: " + " and ".join(what)

        bad_params = [n for n, p in self._params() if not bool(torch.isfinite(p.detach()).all())]
        if bad_params:
            context["bad_params"] = bad_params

        for _, p in self._params():
            p.grad = None
        self._restore(rng, buffers)

        tracer = Tracer(*self.models)
        replay_loss = None
        error = None
        try:
            with tracer:
                replay_loss = fn(*args, **kwargs)
                replay_loss.backward()
        except Exception as e:  # report what we saw before the error
            error = f"{type(e).__name__}: {e}"

        target = None
        if replay_loss is not None and loss_bad:
            target = tracer._lookup(replay_loss.detach()) or tracer._lookup(replay_loss)
        if target is None and bad_grads:
            grads = {n: p.grad for n, p in self._params()}
            for n in bad_grads:
                g = grads.get(n)
                if g is not None and (target := tracer._lookup(g)) is not None:
                    break
        report = tracer.report(target, context)
        if error:
            context["replay_error"] = error
        if not report.found:
            context["not_found"] = (
                "nantrace: the replay did not produce NaN or Inf. The step is not deterministic "
                "(for example nondeterministic GPU kernels or data loading inside the step), "
                "so the bad values could not be traced."
            )

        for _, p in self._params():
            p.grad = None
        self._restore(rng, buffers)
        return report
