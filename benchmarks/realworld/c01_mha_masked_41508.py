# pytorch/pytorch#41508  GT: a fully masked attention row (padding + attn_mask) -> softmax 0/0 -> NaN grads
import torch
torch.manual_seed(0)
attn = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1)
x = torch.rand(3, 2, 1)
key_padding_mask = torch.as_tensor([[False, False, False], [False, True, True]], dtype=torch.bool)
attn_mask = torch.as_tensor([[0., float('-inf'), float('-inf')], [0., 0., float('-inf')], [float('-inf'), 0., 0.]])
output, scores = attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
loss = output[0, :].sum()
loss.backward()
print({n: p.grad for n, p in attn.named_parameters()})
