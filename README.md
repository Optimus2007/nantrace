# nantrace

Your loss went NaN at step 30,000. nantrace tells you **which op created it, on which line, and why**, and suggests the fix.

```
nantrace: step 137: loss is nan and 6 parameter gradients are non-finite (q.weight, q.bias, k.weight ...)

  1. origin forward  aten.exp.default  ->  float32 [16, 8, 8], range [9.345e-42, 4.671e+35], 3 +inf
     at       guard_training.py:21 in forward
              w = scores.exp()
     module   Attention
     input 0  float32 [16, 8, 8], range [-94.47, 2063], all finite
     why      exp overflow: in float32, exp(x) is inf for x above 88.7, and the largest input here is 2063
     fix      subtract the max first (x - x.amax(dim, keepdim=True)), or use torch.softmax, torch.logsumexp or F.log_softmax instead of exp followed by sum

  2. then   forward  aten.div.Tensor  ->  float32 [16, 8, 8], range [0, 1], 3 nan, 13 zeros
     at       guard_training.py:22 in forward
              attn = w / w.sum(-1, keepdim=True)  # softmax without the max shift
     why      inf / inf (or 0 / 0) gives NaN
```

## Why another NaN tool

| | `detect_anomaly` | forward hooks (most NaN tools) | nantrace |
|---|---|---|---|
| Finds NaN created in the forward pass | no, only backward | layer level | op level |
| Finds NaN created in the backward pass | yes, but reports the first bad grad in *reverse* order, often far from the cause | no | yes, mapped to the forward line |
| Follows inf → NaN back to the op that started it | no | no | yes |
| Explains the cause (overflow, log(0), 0/0, fully masked row, float16 range, the `torch.where` gradient trap) | no | no | yes |
| Cost while training | 1.4x to 2x every step | every step | about 0 to 4% (Guard), full tracing only for the bad step |

These are the problems people report about `detect_anomaly` in [pytorch/pytorch#160016](https://github.com/pytorch/pytorch/issues/160016), and the reason JAX users have `jax_debug_nans` while PyTorch users get a backward traceback.

## Use it

### 1. In a training loop: `Guard`

```python
import nantrace

guard = nantrace.Guard(model)
for batch in loader:
    loss = guard.step(compute_loss, batch)   # runs forward and backward
    optimizer.step()
    optimizer.zero_grad()
```

After each backward pass Guard checks the loss and the gradient norms in one fused reduction. When that check fails, it restores the RNG state and module buffers from the start of the step and runs the same step again under the op level tracer. It raises `NonFiniteError` with the report. Use `on_error="skip"` to print the report, clear the gradients so `optimizer.step()` does nothing, and keep training.

The check runs before `optimizer.step()`, so the weights are still the ones that produced the NaN and nothing needs to be copied.

### 2. On a script: `nantrace run`

```bash
nantrace run train.py --steps 50       # stops at the first NaN, exit code 3
nantrace run --json train.py           # machine readable report for agents and CI
nantrace run --stop any train.py       # also stop at the first inf
```

No code changes. Every op is checked, so use it on a short reproduction.

### 3. Around any block: `trace`

```python
with nantrace.trace(model) as tr:
    loss = model(x).logsumexp(-1).mean()
    loss.backward()
print(tr.explain(loss))           # or tr.explain(param.grad)
```

### 4. Before training, in tests: `fuzz`

```python
findings = nantrace.fuzz(my_loss, logits, targets)
assert not findings, "\n\n".join(map(str, findings))
```

Starting from one valid example input, `fuzz` tries edge cases on each argument: zeros, constant values, huge and tiny values, negative values, a batch of one, float16 and bfloat16, masks with a fully masked row, and integer targets that are all one value (such as `ignore_index=-100`). Each try runs forward and backward under the tracer, and every NaN or Inf is reported with its root cause. Put it in a unit test to catch the edge cases users would otherwise report as issues.

From valid inputs, `fuzz` found the bug behind 7 real issues (`benchmarks/realworld/fuzz_rediscover.py`), for example:

```
pytorch#201658 linear_cross_entropy: when arg 2 (integer) is all -100
   origin  inside the custom op: where writes the constant nan, at torch/nn/modules/linear_cross_entropy.py:87
pytorch#201627 gumbel_softmax: when arg 0 is float16 with values up to 25% of the float16 limit
   origin  aten.div at torch/nn/functional.py:2277, float16 overflow (the exact result reaches 1.638e+05)
```

and reported nothing for 5 numerically safe functions, including the fixed version of the TorchRL weighting.

## What it diagnoses

* `exp` overflow, and `exp` underflow that later becomes 0 / 0 (for example softmax weights that all round to 0)
* `log` of 0 or negative values, `log(softmax)` and `log(sigmoid)` underflow
* division by zero, 0 / 0, inf / inf, inf - inf, 0 * inf
* gradients that are infinite at a point: `sqrt`, `log`, `pow`, `acos`, `asin`, `atanh`, norms and distances at 0
* the `torch.where(mask, log(x), 0)` trap, where the forward pass is finite but the gradient is NaN
* softmax over a fully masked row
* float16 and bfloat16 overflow, with the exact value computed again in float64
* NaN that was already in the data or a parameter, named by parameter

## Tested on real bugs

`benchmarks/realworld/` holds the reproducers from 10 public NaN issues (PyTorch, DeepChem, a GNN research repo and others) plus 4 controls that are fixed upstream. `nantrace run` gives the root cause stated in the issue for all 10 and reports nothing for the 4 controls. See `benchmarks/realworld/RESULTS.md`.

## Real bug found with it

Run unchanged on a TorchRL planning script, nantrace pointed to `torchrl/modules/planners/mppi.py:256` and reported that `exp` underflowed to 0 a few lines earlier, so the softmax weights summed to 0. That is the bug in [pytorch/rl#4717](https://github.com/pytorch/rl/issues/4717), which was found by hand before this tool existed.

## Overhead

`benchmarks/overhead.py`: 6 layer MLP, width 1024, batch 256, AdamW, MacBook Air M2.

| | CPU | MPS |
|---|---|---|
| plain training | 1.00x | 1.00x |
| `nantrace.Guard` | 1.04x | about 1.0x (within noise) |
| `torch.autograd.detect_anomaly` | 1.36x | 1.7x to 2.1x |

## Limits

* Not yet tested on CUDA or with multiple GPUs. Reports from CUDA users are very welcome.
* The replay needs the step to be repeatable. Nondeterministic GPU kernels or random data augmentation inside the step can make the NaN disappear on replay; nantrace says so when that happens.
* Code inside `torch.compile` is not traced op by op yet.
* With `GradScaler`, pass `check_grads=False`; inf gradients are expected there and the scaler skips those steps.

## For coding agents

`skills/nantrace/SKILL.md` teaches an agent (Claude Code, Cursor and others) to run `nantrace run --json`, read the origin, and fix the line.

## License

MIT
