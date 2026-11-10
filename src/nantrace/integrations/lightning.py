"""PyTorch Lightning callback.

    from nantrace.integrations.lightning import NantraceCallback

    trainer = L.Trainer(callbacks=[NantraceCallback()])

After every backward pass the callback checks the gradients with one fused
reduction. When one is NaN or Inf, it restores the RNG state and buffers from
the start of the batch, runs `training_step` on the same batch again under the
op level tracer, and raises NonFiniteError with the report (or, with
on_error="skip", prints it and clears the gradients so the optimizer step does
nothing).

With gradient accumulation, the check runs after each micro batch, so the
report points at the micro batch that went wrong. With "16-mixed" precision the
GradScaler expects inf gradients now and then; the callback turns its gradient
check off in that case and you should rely on the scaler.
"""

from __future__ import annotations

try:
    import lightning.pytorch as pl
except ImportError:  # the older package name
    import pytorch_lightning as pl

from ..guard import Guard


class NantraceCallback(pl.Callback):
    def __init__(self, on_error: str = "raise", check_grads: bool | None = None):
        super().__init__()
        self.on_error = on_error
        self.check_grads = check_grads
        self.guard: Guard | None = None
        self._batch = None

    def setup(self, trainer, pl_module, stage=None):
        check = self.check_grads
        if check is None:
            check = "16-mixed" not in str(getattr(trainer, "precision", ""))
        self.guard = Guard(pl_module, check_grads=check, on_error=self.on_error)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.guard.begin()
        self._batch = (batch, batch_idx)

    def on_after_backward(self, trainer, pl_module):
        _, bad = self.guard.check()
        if not bad:
            return
        batch, batch_idx = self._batch

        def recompute():
            out = pl_module.training_step(batch, batch_idx)
            return out["loss"] if isinstance(out, dict) else out

        self.guard.handle(recompute, bad_grads=bad)
