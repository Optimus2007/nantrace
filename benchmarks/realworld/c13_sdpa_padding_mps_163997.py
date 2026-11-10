# pytorch/pytorch#163997 on MPS (original report device)  GT: all -inf mask rows -> NaN rows
import torch
import torch.nn.functional as F
torch.manual_seed(0)
B, H, N, C = 1, 2, 6, 8
q, k, v = (torch.randn(B, H, N, C, device="mps") for _ in range(3))
valid = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool, device="mps")
allowed = valid[:, None] & valid[None, :]
attn_mask = torch.zeros(B, 1, N, N, device="mps").masked_fill(~allowed, float("-inf"))
x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
print(x[0, 0, -1])
