"""nantrace: find where NaN and Inf start in PyTorch, and why."""

from contextlib import contextmanager

from .fuzz import Finding, fuzz
from .guard import Guard
from .tracer import Event, NonFiniteError, Report, Tracer

__all__ = ["Guard", "Tracer", "Report", "Event", "NonFiniteError", "trace", "fuzz", "Finding"]
__version__ = "0.1.0"


@contextmanager
def trace(*models, stop=None):
    """Trace every op in the block. Use `tracer.explain(tensor)` or `tracer.report()` after."""
    tracer = Tracer(*models, stop=stop)
    with tracer:
        yield tracer
