# The forward pass is finite, but the gradient is NaN.
# torch.where still differentiates log(x) at the masked entries, and the gradient of log at 0 is 0/0.
import torch

x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
y = torch.where(x > 0, torch.log(x), torch.zeros_like(x))
y.sum().backward()
print("forward", y.tolist(), "grad", x.grad.tolist())
