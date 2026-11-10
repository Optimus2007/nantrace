---
name: nantrace
description: Find and fix the root cause of NaN or Inf in PyTorch code, and test losses or layers on edge case inputs. Use when a loss, gradient or output becomes nan or inf, when training diverges, when a user reports "loss is nan", or when writing or reviewing numerically delicate code (softmax, log, exp, division, masking, float16).
---

# Fixing NaN and Inf in PyTorch with nantrace

If the nantrace MCP tools (`trace_nan`, `fuzz`) are available, use them. Otherwise use the
command line shown next to each step. Both give the same short JSON.

## When training or a script produces NaN

1. Make a short reproduction: the smallest script (or a few training steps) that still produces the NaN. Fix the seed. Tracing checks every op, so keep it small.
2. Trace it: `trace_nan(script="repro.py")`, or `nantrace run --brief repro.py`.
   * `exit_code` 3 means a NaN was found. 4 means no NaN, but an Inf that is still alive at the end, such as an infinite loss. 0 means nothing was found.
   * `cause` is the first step that is a real bug. It skips `-inf` masks that were written on purpose.
   * `cause.location` is the line to change. For `phase: "backward"` it is the forward line whose gradient blew up.
   * `cause.inside_library` is the line inside torch or another library, when the bug is there. `cause.inside_custom_op` goes one level deeper for custom ops.
   * `cause.why` explains it and `cause.fix` suggests the change. `cause.inputs` shows the value ranges that triggered it.
   * `chain` shows how the value then spread. Fix the cause, not the later steps.
3. Apply the fix at the cause. Prefer the numerically stable builtin (`torch.softmax`, `torch.logsumexp`, `F.log_softmax`, `F.logsigmoid`, `F.softplus`, `F.binary_cross_entropy_with_logits`, adding eps before `sqrt`, `log` or a division) over clamping the final loss or calling `nan_to_num`, which hides the problem.
4. Trace again and confirm `found` is false.

If `cause.phase` is `"input"`, the NaN was already in the data or in a parameter before the traced code ran. Look at the data pipeline or the previous optimizer step.

If the NaN only appears after many steps, add `nantrace.Guard(model)` to the training loop instead (see the README). It costs almost nothing per step and reports the bad step when it happens.

## Before training, or after a fix: fuzz

Write a small file with `fuzz_` functions. Each returns the function or module to test and one valid example input:

```python
import torch
from mymodel import my_loss

def fuzz_my_loss():
    return my_loss, (torch.randn(8, 10), torch.randint(0, 10, (8,)))
```

Run `fuzz(path="fuzz_checks.py")`, or `nantrace fuzz --json fuzz_checks.py`. Each finding says which edge case broke it (`when`), for example "arg 1 (integer) is all -100" or "arg 0 is float16 with values up to 25% of the float16 limit", plus the same `cause` fields as above. Not every finding needs a fix: decide whether that input can really happen. Adding the fuzz call to a unit test keeps the bug from coming back:

```python
def test_my_loss_is_finite():
    assert not nantrace.fuzz(my_loss, torch.randn(8, 10), torch.randint(0, 10, (8,)))
```
