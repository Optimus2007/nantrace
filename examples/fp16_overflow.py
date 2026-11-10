# float16 matmul whose result exceeds 65504.
import torch

a = torch.full((8, 64), 40.0, dtype=torch.float16)
b = torch.full((64, 8), 40.0, dtype=torch.float16)
h = a @ b                      # 64 * 40 * 40 = 102400 > 65504
out = h - h.mean()
print(out.float().mean())
