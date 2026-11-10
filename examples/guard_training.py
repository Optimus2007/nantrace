# A training loop that runs fine for a while, then one batch with an outlier
# makes the hand written softmax overflow. Guard costs almost nothing per step
# and only traces the step that went wrong.
import torch
import torch.nn.functional as F

import nantrace

torch.manual_seed(0)


class Attention(torch.nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.q = torch.nn.Linear(d, d)
        self.k = torch.nn.Linear(d, d)
        self.out = torch.nn.Linear(d, 1)

    def forward(self, x):
        scores = self.q(x) @ self.k(x).transpose(-1, -2)
        w = scores.exp()
        attn = w / w.sum(-1, keepdim=True)  # softmax without the max shift
        return self.out(attn @ x).squeeze(-1)


model = Attention()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
guard = nantrace.Guard(model)

for step in range(1, 201):
    x = torch.randn(16, 8, 32)
    if step == 137:
        x[3, 2] *= 50  # one corrupted sample
    target = torch.randn(16, 8)
    loss = guard.step(lambda: F.mse_loss(model(x), target))
    opt.step()
    opt.zero_grad()
    if step % 50 == 0:
        print(f"step {step} loss {loss.item():.3f}")
