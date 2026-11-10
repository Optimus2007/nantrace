"""Small summaries of tensor values, used in reports and diagnosis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

# Largest finite value for each floating dtype.
FMAX = {
    torch.float16: 65504.0,
    torch.bfloat16: 3.3895e38,
    torch.float32: 3.4028e38,
    torch.float64: 1.7977e308,
}


def is_float(t: object) -> bool:
    return isinstance(t, torch.Tensor) and (t.is_floating_point() or t.is_complex())


def has_nonfinite(t: torch.Tensor) -> bool:
    """True if a floating tensor holds at least one NaN or Inf."""
    if t.numel() == 0:
        return False
    with torch.no_grad():
        return not bool(torch.isfinite(t.detach()).all())


@dataclass
class TensorStats:
    shape: tuple
    dtype: str
    numel: int
    nan: int
    posinf: int
    neginf: int
    min: float | None  # over finite values only
    max: float | None
    zeros: int
    negatives: int
    min_abs_nonzero: float | None

    @property
    def finite(self) -> bool:
        return self.nan == 0 and self.posinf == 0 and self.neginf == 0

    @property
    def absmax(self) -> float | None:
        if self.min is None:
            return None
        return max(abs(self.min), abs(self.max))

    def short(self) -> str:
        parts = [f"{self.dtype} {list(self.shape)}"]
        if self.min is not None:
            parts.append(f"range [{self.min:.4g}, {self.max:.4g}]")
        bad = []
        if self.nan:
            bad.append(f"{self.nan} nan")
        if self.posinf:
            bad.append(f"{self.posinf} +inf")
        if self.neginf:
            bad.append(f"{self.neginf} -inf")
        parts.append(", ".join(bad) if bad else "all finite")
        if self.zeros:
            parts.append(f"{self.zeros} zeros")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


def stats(t: torch.Tensor) -> TensorStats:
    with torch.no_grad():
        x = t.detach()
        if x.is_complex():
            x = torch.view_as_real(x)
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        n = x.numel()
        isnan = torch.isnan(x)
        posinf = x == float("inf")
        neginf = x == float("-inf")
        fin = x[torch.isfinite(x)]
        nz = fin[fin != 0].abs()
        return TensorStats(
            shape=tuple(t.shape),
            dtype=str(t.dtype).replace("torch.", ""),
            numel=n,
            nan=int(isnan.sum()),
            posinf=int(posinf.sum()),
            neginf=int(neginf.sum()),
            min=float(fin.min()) if fin.numel() else None,
            max=float(fin.max()) if fin.numel() else None,
            zeros=int((fin == 0).sum()),
            negatives=int((fin < 0).sum()),
            min_abs_nonzero=float(nz.min()) if nz.numel() else None,
        )
