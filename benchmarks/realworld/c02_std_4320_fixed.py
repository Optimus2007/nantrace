# pytorch/pytorch#4320  NEGATIVE CONTROL: fixed upstream, expect no NaN
import torch
a = torch.zeros(16, 3, 64, 64, requires_grad=True)
a.std(dim=0).mean().backward()
print(a.grad.abs().max())
