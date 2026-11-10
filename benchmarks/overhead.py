"""Per step time: plain training vs nantrace.Guard vs torch.autograd.detect_anomaly."""
import sys
import time
import warnings

import torch

import nantrace

warnings.filterwarnings("ignore")
device = sys.argv[1] if len(sys.argv) > 1 else "cpu"


def make():
    torch.manual_seed(0)
    layers = []
    for _ in range(6):
        layers += [torch.nn.Linear(1024, 1024), torch.nn.LayerNorm(1024), torch.nn.GELU()]
    model = torch.nn.Sequential(*layers, torch.nn.Linear(1024, 10)).to(device)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-4)


x = torch.randn(256, 1024, device=device)
y = torch.randint(0, 10, (256,), device=device)


def sync():
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def bench(mode, steps=30):
    model, opt = make()
    guard = nantrace.Guard(model)
    loss_fn = lambda: torch.nn.functional.cross_entropy(model(x), y)
    for i in range(steps + 5):
        if i == 5:
            sync()
            t = time.perf_counter()
        if mode == "guard":
            guard.step(loss_fn)
        elif mode == "anomaly":
            with torch.autograd.detect_anomaly():
                loss_fn().backward()
        else:
            loss = loss_fn()
            loss.backward()
            loss.item()  # most loops log the loss anyway
        opt.step()
        opt.zero_grad()
    sync()
    return (time.perf_counter() - t) / steps * 1000


base = bench("plain")
for mode in ("plain", "guard", "anomaly"):
    ms = bench(mode) if mode != "plain" else base
    print(f"{device:4} {mode:8} {ms:7.2f} ms/step  x{ms / base:.2f}")
