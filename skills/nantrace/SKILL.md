---
name: nantrace
description: Find and fix the root cause of NaN or Inf in PyTorch code. Use when a loss, gradient, or output becomes nan or inf, when training diverges, or when a user reports "loss is nan".
---

# Fixing NaN and Inf in PyTorch with nantrace

1. Make a short reproduction: the smallest script (or a few training steps) that still produces the NaN. Fix the seed.
2. Run it under nantrace and read the JSON:

   ```bash
   nantrace run --json repro.py
   ```

   Exit code 3 means a NaN was found. Exit code 4 means no NaN, but an Inf that is still alive at the end, such as an infinite loss. Read `cause`, the first step that is a real bug (it skips `-inf` masks that were written on purpose). `origin` is the first op in the chain:
   * `cause.location.file` and `cause.location.line` give the line to change, and `cause.inner` gives the line inside a library such as torch, when there is one. For `phase: "backward"` this is the forward line whose gradient blew up.
   * `cause.cause` explains why. `cause.fix` is the suggested change.
   * `cause.inputs` show the value ranges that triggered it.
   * `chain` lists how the value then spread into NaN. Fix the origin, not the later steps.
3. Apply the fix at the origin line. Prefer the numerically stable builtin (`torch.softmax`, `torch.logsumexp`, `F.log_softmax`, `F.logsigmoid`, `F.softplus`, adding eps before `sqrt` or `log`) over clamping the final loss or calling `nan_to_num`, which hides the problem.
4. Run `nantrace run repro.py` again and confirm it reports no NaN.

If the bug only shows up after many steps, add `nantrace.Guard(model)` to the training loop instead (see the README). It costs almost nothing per step and reports the bad step when it happens.

If `cause.phase` is `"input"`, the NaN was already in the data or in a parameter before the traced code ran. Look at the data pipeline or the previous optimizer step.
