"""Hugging Face Trainer with nantrace built in.

    from nantrace.integrations.hf import NantraceTrainer

    trainer = NantraceTrainer(model=model, args=args, train_dataset=ds)   # same arguments as Trainer
    trainer.train()

After every training step (every micro batch with gradient accumulation) it
checks the loss and the gradients with one fused reduction. When one is NaN or
Inf, it restores the RNG state and buffers, computes the loss of the same inputs
again under the op level tracer, and raises NonFiniteError with the report.
Pass nantrace_on_error="skip" to print the report, clear the gradients and keep
training.
"""

from __future__ import annotations

from transformers import Trainer

from ..guard import Guard


class NantraceTrainer(Trainer):
    def __init__(self, *args, nantrace_on_error: str = "raise", nantrace_check_grads: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._nantrace_on_error = nantrace_on_error
        self._nantrace_check_grads = nantrace_check_grads
        self.nantrace_guard: Guard | None = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.nantrace_guard is None:
            check = self._nantrace_check_grads
            if check is None:  # fp16 with a GradScaler expects inf gradients now and then
                check = not self.args.fp16
            self.nantrace_guard = Guard(model, check_grads=check, on_error=self._nantrace_on_error)
        guard = self.nantrace_guard
        guard.begin()
        loss = super().training_step(model, inputs, num_items_in_batch)
        loss_bad, bad = guard.check(loss)
        if loss_bad or bad:

            def recompute():
                model.train()
                with self.compute_loss_context_manager():
                    return self.compute_loss(model, self._prepare_inputs(inputs), num_items_in_batch=num_items_in_batch)

            guard.handle(recompute, loss=loss, loss_bad=loss_bad, bad_grads=bad)
        return loss
