# GraphBerry/DiG-In-GNN#5  GT: mean over an empty index list (batch without positive nodes) -> NaN
import torch
feats = torch.randn(4, 3, requires_grad=True)
classify_label_1 = []
positive_feats = torch.mean(feats[classify_label_1], dim=0, keepdim=True)
loss = (feats - positive_feats).pow(2).mean()
loss.backward()
print(loss)
