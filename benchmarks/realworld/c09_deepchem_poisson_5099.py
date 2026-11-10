# deepchem/deepchem#5099  GT: torch.log(output) with output == 0 (PoissonLoss, deepchem/models/losses.py:158)
import torch
def loss_fn(output, labels):  # verbatim from deepchem PoissonLoss._create_pytorch_loss
    return torch.mean(output - labels * torch.log(output))
output = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
labels = torch.tensor([1.0, 1.0, 1.0])
loss = loss_fn(output, labels)
loss.backward()
print(loss, output.grad)
