# pytorch/pytorch#201272  GT: zero rate -> 0 * log(0) = 0 * -inf = NaN inside the Poisson KL formula
import torch
from torch.distributions import Poisson, kl_divergence
p = Poisson(torch.tensor([0.0, 0.0, 1.0]))
q = Poisson(torch.tensor([0.0, 1.0, 0.0]))
print(kl_divergence(p, q))
