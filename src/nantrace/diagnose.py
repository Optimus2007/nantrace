"""Explain why an op produced NaN or Inf, and suggest a fix.

Each rule looks at the op name, the phase (forward or backward) and summaries of
the inputs. Rules are written for the common causes seen in real training code.
"""

from __future__ import annotations

import math

from ._stats import FMAX, TensorStats

EPS_FIX = "add a small eps (for example 1e-8) or clamp the input"


def _fmax(dtype: str) -> float:
    for k, v in FMAX.items():
        if str(k).replace("torch.", "") == dtype:
            return v
    return 3.4028e38  # float32


def _first(inputs: list[TensorStats]) -> TensorStats | None:
    return inputs[0] if inputs else None


def diagnose(
    op: str,
    phase: str,
    inputs: list[TensorStats],
    out: TensorStats,
    grad_fn: str | None = None,
    finite_in_float64: bool | None = None,
    float64_absmax: float | None = None,
    constants: list[float] | None = None,
) -> tuple[str, str]:
    """Return (cause, fix) as plain sentences."""
    name = op.split(".")[1] if op.startswith("aten.") else op
    name = name.rstrip("_")
    x = _first(inputs)
    dt = out.dtype
    lim = _fmax(dt)
    makes_nan = out.nan > 0
    inputs_nonfinite = any(not s.finite for s in inputs)
    bw = phase == "backward"
    where = f" while computing a gradient ({grad_fn})" if bw and grad_fn else ""

    # The op was handed a NaN or inf constant, for example masked_fill(mask, -inf) or where(c, nan, x).
    if constants and not inputs_nonfinite:
        vals = ", ".join(sorted({str(v) for v in constants}))
        if any(math.isnan(v) for v in constants):
            return (
                f"{name} writes the constant {vals} into the tensor (it is passed in as an argument){where}",
                "if NaN is meant as a marker, make sure masked or ignored entries never reach the loss or "
                "its gradient; otherwise use a finite value",
            )
        return (
            f"{name} writes the constant {vals} into the tensor (usually a mask){where}",
            "fine if every row keeps at least one finite entry; otherwise use torch.finfo(dtype).min",
        )

    # Backward origins: the autograd node says which forward op this gradient belongs to.
    if bw and grad_fn and not inputs_nonfinite:
        hit = _backward_rule(grad_fn, makes_nan)
        if hit:
            return hit

    # Non-finite values flowing in: explain how they became NaN here.
    if inputs_nonfinite and makes_nan:
        if name in ("_softmax", "softmax", "_log_softmax", "log_softmax") and x and x.posinf:
            return (
                f"softmax got +inf{where}; it subtracts the row max, and inf - inf is NaN",
                "",
            )
        if name in ("_softmax", "softmax", "_log_softmax", "log_softmax") and x and x.neginf:
            return (
                f"softmax got a row where every entry is -inf (a fully masked row){where}, so it computes 0/0",
                "keep at least one unmasked position per row, or fill masks with torch.finfo(dtype).min instead of -inf",
            )
        if "scaled_dot_product" in name or "attention" in name:
            if any(s.neginf for s in inputs):
                return (
                    f"attention got a mask where some query row has every key masked (-inf){where}, "
                    "so its softmax is 0/0",
                    "make sure every query attends to at least one key, or fill masks with torch.finfo(dtype).min, "
                    "and zero those rows afterwards",
                )
        if name in ("mul", "addcmul") or name.startswith("mul"):
            return (f"0 * inf gives NaN{where}", "")
        if name in ("sub", "add", "rsub") or name.startswith("add"):
            return (f"inf - inf gives NaN{where}", "")
        if name in ("div",):
            return (f"inf / inf (or 0 / 0) gives NaN{where}", "")
        return (f"{name} turned an incoming inf into NaN{where}", "")

    # Origins: all inputs finite, output not.
    if name in ("exp", "exp2", "expm1", "sinh", "cosh"):
        thr = math.log(lim)
        mx = x.max if x else None
        return (
            f"exp overflow{where}: in {dt}, exp(x) is inf for x above {thr:.1f}, and the largest input here is {mx:.4g}"
            if mx is not None
            else f"exp overflow{where}",
            "subtract the max first (x - x.amax(dim, keepdim=True)), or use torch.softmax, "
            "torch.logsumexp or F.log_softmax instead of exp followed by sum",
        )
    if name in ("log", "log2", "log10"):
        if bw and x and x.zeros:
            return (
                f"the gradient of log at 0 is 1/0{where} ({x.zeros} zero inputs)",
                "clamp the input: torch.log(x.clamp_min(eps)); for probabilities use F.log_softmax or F.logsigmoid",
            )
        if x and x.negatives:
            return (
                f"log of negative numbers is NaN{where} ({x.negatives} negative inputs, min {x.min:.4g})",
                "check why the input can be negative; clamp with x.clamp_min(eps) if it must be positive",
            )
        if x and x.zeros:
            return (
                f"log(0) is -inf{where} ({x.zeros} inputs are exactly 0, often a probability that underflowed)",
                "for probabilities use F.log_softmax / F.logsigmoid instead of log(softmax) / log(sigmoid); "
                "otherwise torch.log(x.clamp_min(eps))",
            )
    if name == "log1p" and x and x.min is not None and x.min <= -1:
        return (f"log1p(x) is undefined for x <= -1{where} (min input {x.min:.4g})", "clamp x above -1")
    if name in ("div", "reciprocal"):
        # div with a Python number divisor has a single tensor input: the numerator
        den = inputs[1] if name == "div" and len(inputs) >= 2 else (x if name == "reciprocal" else None)
        num = inputs[0] if name == "div" and len(inputs) >= 2 else None
        if den and den.zeros:
            if num is not None and num.zeros:
                return (
                    f"0 / 0{where}: the denominator has {den.zeros} zeros and so does the numerator",
                    "add eps to the denominator, or guard empty sums and counts (for example a mask with no True entries)",
                )
            return (
                f"division by zero{where}: the denominator has {den.zeros} zeros",
                "add eps to the denominator: x / (d + eps), or d.clamp_min(eps)",
            )
        if den and den.min_abs_nonzero is not None:
            return (
                f"division by a tiny number overflowed {dt}{where} (smallest |denominator| {den.min_abs_nonzero:.3g})",
                "add eps to the denominator, or compute in log space",
            )
    if name in ("sqrt", "rsqrt", "pow") and x:
        if x.negatives and name != "pow":
            return (f"{name} of negative numbers is NaN{where} ({x.negatives} negative inputs)", "clamp the input to >= 0")
        if x.zeros and (bw or name == "rsqrt"):
            return (
                f"the gradient of sqrt at 0 is infinite{where} ({x.zeros} zero inputs); "
                "this often comes from torch.norm or a distance between identical vectors",
                "use sqrt(x + eps), or (x.pow(2).sum(dim) + eps).sqrt() instead of norm",
            )
        if name == "pow" and not finite_in_float64:
            if x.negatives and makes_nan:
                return (f"a negative number to a fractional power is NaN{where}", "clamp the base to >= 0, or use abs()")
            if x.zeros:
                return (f"0 to a negative power is inf{where}", EPS_FIX)
    if name in ("acos", "asin", "atanh"):
        return (
            f"{name} is undefined outside [-1, 1] and its gradient is infinite at +-1{where}",
            "clamp the input to [-1 + eps, 1 - eps]",
        )
    if name in ("linalg_vector_norm", "norm", "_euclidean_dist", "cdist") and bw:
        return (
            f"the gradient of a norm at the zero vector is 0/0{where}",
            "add eps inside the square root: (x.pow(2).sum(dim) + eps).sqrt()",
        )
    if name in ("var", "std", "var_mean", "std_mean"):
        return (
            f"variance with only one sample divides by zero{where} (correction=1 needs at least 2 values)",
            "use correction=0 or make sure the reduced dimension has more than one element",
        )
    if name in ("mean",) and x and x.numel == 0:
        return ("mean of an empty tensor is NaN", "check for empty batches or masks with no True entries")
    if name in ("_to_copy", "copy") and dt in ("float16", "bfloat16") and x and x.absmax is not None:
        return (
            f"casting to {dt} overflowed: {x.absmax:.4g} is beyond the {dt} limit of {lim:.5g}",
            "keep this tensor in float32, or rescale it before casting",
        )

    if finite_in_float64:
        big = f" (the exact result reaches {float64_absmax:.4g})" if float64_absmax is not None else ""
        matmul = name in ("mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "convolution", "_convolution")
        if dt == "float16":
            return (
                f"float16 overflow in {name}{where}: the result does not fit in float16 (max 65504){big}",
                "use bfloat16, keep this op in float32 (autocast does this for some ops), "
                "or scale activations down with normalization",
            )
        what = "matrix product" if matmul else name
        return (
            f"{what} overflowed {dt}{where}{big}; inputs are finite but too large",
            "reduce the scale: lower the learning rate, clip gradients, normalize activations, or check the initialization",
        )
    if finite_in_float64 is False:
        return (f"{name} is undefined for these inputs{where}", "inspect the input ranges below")
    return (f"{name} produced non-finite values from finite inputs{where}", "inspect the input ranges below")


_MASKED_NOTE = (
    "The gradient is 0/0, which usually means this value was masked out or unused "
    "(torch.where, masked_fill, multiplying by 0), but autograd still differentiates it. "
)
_BACKWARD = {
    # grad_fn prefix: (forward op, why it blows up, fix)
    "Log": (
        "log",
        "the gradient of log(x) is 1/x, which is infinite at x = 0",
        "make the input safe before log, not after: log(torch.where(mask, x, 1)) instead of "
        "torch.where(mask, log(x), 0); or use x.clamp_min(eps)",
    ),
    "Sqrt": (
        "sqrt",
        "the gradient of sqrt(x) is 1/(2 sqrt(x)), which is infinite at x = 0; this often comes from a "
        "distance or norm between identical vectors",
        "use sqrt(x + eps), or for distances (diff.pow(2).sum(dim) + eps).sqrt()",
    ),
    "Rsqrt": ("rsqrt", "the gradient of rsqrt(x) is infinite at x = 0", "use rsqrt(x + eps)"),
    "Pow": (
        "pow",
        "the gradient of x ** p with p < 1 is infinite at x = 0",
        "add eps to the base: (x + eps) ** p",
    ),
    "Div": (
        "div",
        "the gradient of a / b is infinite where b = 0",
        "add eps to the denominator in the forward pass: a / (b + eps)",
    ),
    "Acos": ("acos", "the gradient of acos is infinite at +-1", "clamp the input to [-1 + eps, 1 - eps]"),
    "Asin": ("asin", "the gradient of asin is infinite at +-1", "clamp the input to [-1 + eps, 1 - eps]"),
    "Atanh": ("atanh", "the gradient of atanh is infinite at +-1", "clamp the input to [-1 + eps, 1 - eps]"),
    "Norm": (
        "norm",
        "the gradient of a norm is 0/0 at the zero vector",
        "add eps inside the square root: (x.pow(2).sum(dim) + eps).sqrt()",
    ),
    "LinalgVectorNorm": (
        "norm",
        "the gradient of a norm is 0/0 at the zero vector",
        "add eps inside the square root: (x.pow(2).sum(dim) + eps).sqrt()",
    ),
    "Cdist": ("cdist", "the gradient of cdist is undefined for identical points", "add eps to the squared distance before sqrt"),
}


def _backward_rule(grad_fn: str, makes_nan: bool) -> tuple[str, str] | None:
    for prefix, (fwd, why, fix) in _BACKWARD.items():
        if grad_fn.startswith(prefix + "Backward"):
            cause = f"in the backward pass of {fwd} ({grad_fn}): {why}."
            if makes_nan and fwd in ("log", "sqrt", "div", "pow"):
                cause += " " + _MASKED_NOTE.strip()
            return cause, fix
    return None
