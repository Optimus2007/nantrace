# pytorch/pytorch#201627  GT: (logits + gumbels) / tau overflows float16 (> 65504) inside gumbel_softmax -> softmax NaN
import torch
import torch.nn.functional as F
logits = torch.tensor([[10000.0, -10000.0, 0.5, -0.5]], dtype=torch.float16)
torch.manual_seed(0)
out = F.gumbel_softmax(logits, tau=0.1, hard=True)
print(out)
