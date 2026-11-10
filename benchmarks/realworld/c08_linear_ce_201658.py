# pytorch/pytorch#201658  GT: _neg_weight_target divides by torch.where(d == 0, nan, -d); all targets ignored -> NaN grads
import torch
import torch.nn.functional as F
from torch.nn import LinearCrossEntropyOptions
torch.manual_seed(0)
x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
w = torch.randn(16, 4, dtype=torch.float64, requires_grad=True)
target = torch.full((8,), -100)
loss = F.linear_cross_entropy(x, w, target, options=LinearCrossEntropyOptions())
loss.backward()
print(loss, x.grad.isnan().any())
