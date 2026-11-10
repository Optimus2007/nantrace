# Hand written softmax over large logits: exp overflows to inf, then inf / inf is NaN.
import torch

torch.manual_seed(0)
model = torch.nn.Linear(16, 8)
x = torch.randn(4, 16) * 400

logits = model(x)
w = torch.exp(logits)
probs = w / w.sum(-1, keepdim=True)
loss = -(probs[:, 0] + 1e-9).log().mean()
loss.backward()
print("loss", loss.item())
