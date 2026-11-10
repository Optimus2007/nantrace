import torch, torch.nn.functional as F
import nantrace
torch.manual_seed(0)
safe = {
    "MPPI weights with max shift (the #4717 fix)": (lambda v, a: (torch.softmax(v, 0) * a).sum(0), (torch.randn(8, 1), torch.randn(8, 1))),
    "log_softmax NLL": (lambda l: -F.log_softmax(l, -1)[:, 0].mean(), (torch.randn(4, 5),)),
    "BCE with logits": (lambda l, y: F.binary_cross_entropy_with_logits(l, y), (torch.randn(6), torch.rand(6))),
    "logsigmoid": (lambda x: F.logsigmoid(x).sum(), (torch.randn(5),)),
    "LayerNorm + Linear module": (torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4)), (torch.randn(3, 4),)),
}
for name, (fn, args) in safe.items():
    found = nantrace.fuzz(fn, *args)
    print(f"{name}: {len(found)} finding(s)")
    for f in found:
        e = f.report.cause_event
        print("   ", f.mutation, "->", e.op, "|", e.cause[:140])
