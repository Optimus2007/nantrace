# Start every function from a VALID input that gives finite results, and see if fuzz rediscovers the real bug.
import torch, torch.nn.functional as F
from torch.distributions import Poisson, kl_divergence
from torch.nn import LinearCrossEntropyOptions
import nantrace

torch.manual_seed(0)
cases = {
    "pytorch#201272 Poisson KL": (lambda r1, r2: kl_divergence(Poisson(r1), Poisson(r2)), (torch.rand(4) + 0.5, torch.rand(4) + 0.5)),
    "pytorch#201627 gumbel_softmax": (lambda l: F.gumbel_softmax(l, tau=0.1, hard=True), (torch.randn(2, 4) * 50,)),
    "pytorch#201658 linear_cross_entropy": (lambda x, w, t: F.linear_cross_entropy(x, w, t, options=LinearCrossEntropyOptions()),
        (torch.randn(8, 4, dtype=torch.float64), torch.randn(16, 4, dtype=torch.float64), torch.randint(0, 16, (8,)))),
    "deepchem#5099 PoissonLoss": (lambda out, lab: torch.mean(out - lab * torch.log(out)), (torch.rand(3) + 0.5, torch.ones(3))),
    "DiG-In-GNN#5 empty class mean": (lambda f, lbl: (f - f[lbl.bool()].mean(0, keepdim=True)).pow(2).mean(), (torch.randn(6, 3), torch.tensor([1, 0, 1, 0, 1, 0]))),
    "pytorch#41508 MHA masks": (None, None),
    "torchrl#4717 MPPI weights": (lambda vals, acts: ((torch.exp(vals) * acts).sum(0) / torch.exp(vals).sum(0)), (torch.randn(8, 1), torch.randn(8, 1))),
}
mha = torch.nn.MultiheadAttention(4, 1, batch_first=True)
cases["pytorch#41508 MHA masks"] = (lambda x, kpm: mha(x, x, x, key_padding_mask=kpm)[0], (torch.randn(2, 3, 4), torch.tensor([[False, False, True], [False, True, True]])))

for name, (fn, args) in cases.items():
    found = nantrace.fuzz(fn, *args)
    print(f"## {name}: {len(found)} finding(s)")
    for f in found:
        o = f.report.cause_event
        loc = o.inner or o.location
        print(f"   when {f.mutation}{' (+%d more)' % len(f.also) if f.also else ''} -> {f.where}")
        print(f"      origin {o.op} at {loc.short() if loc else '?'}")
        print(f"      why {o.cause[:150]}")
