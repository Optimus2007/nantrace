# Real bug benchmark

Each `c*.py` is the reproducer from a public issue, copied as written (old `Variable` code updated to current PyTorch).
The first line of each file says the root cause that the issue discussion reached.
Run: `python score.py $(which nantrace) .`

Results on 2026-11-10, torch 2.15.0, macOS arm64 (CPU, and MPS for c13):

| Case | Issue | Root cause from the issue | nantrace origin and cause | Correct |
|---|---|---|---|---|
| c01 | pytorch/pytorch#41508 | fully masked attention row in MultiheadAttention gives NaN grads | `_softmax`: row where every entry is -inf | yes |
| c04 | pytorch/pytorch#10729 | exp overflow, then clamp backward 0 * inf | `exp` overflow at line 4, then 0 * inf in ExpBackward0 | yes |
| c05 | pytorch/pytorch#15506 | log(0) at an unused element, NaN gradient | LogBackward0, 0/0 at a masked or unused value | yes |
| c06 | pytorch/pytorch#201627 | (logits + gumbels) / tau overflows float16 | `div` float16 overflow (exact 1e5), then softmax of +inf | yes |
| c07 | pytorch/pytorch#201272 | 0 * log(0) in the Poisson KL | `log(0)` inside torch/distributions/kl.py:493, then `inf - inf` | yes |
| c08 | pytorch/pytorch#201658 | `torch.where(d == 0, nan, -d)` in `_neg_weight_target` | inside the custom op: `where` writes nan, at linear_cross_entropy.py:87 | yes |
| c09 | deepchem/deepchem#5099 | `torch.log(output)` with output 0 | `log(0)` at the loss line (exit 4: inf, no NaN) | yes |
| c10 | GraphBerry/DiG-In-GNN#5 | mean over an empty index list | `mean` of an empty tensor | yes |
| c12 | common pattern | hand written BCE, `log(1 - sigmoid)` | `log(0)`, then 0 * inf | yes |
| c02 | pytorch/pytorch#4320 | fixed upstream (negative control) | no report | yes |
| c03 | pytorch/pytorch#3264 | fixed upstream (negative control) | no report | yes |
| c11 | pytorch/pytorch#163997 (CPU) | returns zeros on torch 2.15 CPU (negative control) | no report | yes |
| c13 | pytorch/pytorch#163997 (MPS) | returns zeros on torch 2.15 MPS (negative control) | no report | yes |

9 of 9 bugs (8 public issues and 1 common pattern) found with the right root cause; 0 false reports on 4 controls.

Bugs this benchmark found in nantrace itself, now fixed: a Python number divisor was taken as the denominator (c06);
inf only losses were reported as "no NaN" (c09); custom ops were opaque (c08); a NaN in the unused attention weights
output on MPS was reported (c13); a temporary inf that PyTorch masks inside the std gradient was reported (c02).
