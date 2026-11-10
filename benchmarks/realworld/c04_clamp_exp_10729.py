# pytorch/pytorch#10729  GT: exp(90), exp(100) overflow to inf in forward; clamp backward then gives 0 * inf = NaN
import torch
x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True)
y = torch.exp(x)
z = y.clamp(max=1)
z.sum().backward()
print(x.grad)
