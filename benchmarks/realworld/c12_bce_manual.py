# Common pattern (StackOverflow staple)  GT: hand written BCE log(1 - sigmoid(x)) -> log(0) for large logits
import torch
torch.manual_seed(0)
model = torch.nn.Linear(10, 1)
x = torch.randn(32, 10) * 30
y = (torch.rand(32, 1) > 0.5).float()
p = torch.sigmoid(model(x))
loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
loss.backward()
print(loss)
