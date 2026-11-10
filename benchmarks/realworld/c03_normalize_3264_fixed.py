# pytorch/pytorch#3264  NEGATIVE CONTROL: fixed upstream, expect no NaN
import torch
import torch.nn.functional as F
a = torch.tensor([[0, 0, 0, 0, 0.1996]], requires_grad=True)
F.normalize(a, p=1, dim=1).sum().backward()
print(a.grad)
