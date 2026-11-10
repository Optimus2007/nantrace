# pytorch/pytorch#15506  GT: log(0) at an element not used by the loss; LogBackward computes 0/0 -> NaN grad
import torch
a = torch.arange(4, dtype=torch.float, requires_grad=True)
a.log()[-1].backward()
print(a.grad)
