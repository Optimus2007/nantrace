# Run with: nantrace fuzz examples/fuzz_checks.py
# Each fuzz_ function returns the function (or nn.Module) to test and one valid example input.
import torch
import torch.nn.functional as F


def manual_softmax_weights(values, actions):
    w = values.exp()  # no max shift: overflows for large values
    return (w * actions).sum(0) / w.sum(0)


def stable_softmax_weights(values, actions):
    return (torch.softmax(values, 0) * actions).sum(0)


def class_mean_distance(feats, labels):
    positive = feats[labels.bool()].mean(0, keepdim=True)  # empty when no label is 1
    return (feats - positive).pow(2).mean()


def fuzz_manual_softmax_weights():
    return manual_softmax_weights, (torch.randn(8, 1), torch.randn(8, 1))


def fuzz_stable_softmax_weights():
    return stable_softmax_weights, (torch.randn(8, 1), torch.randn(8, 1))


def fuzz_class_mean_distance():
    return class_mean_distance, (torch.randn(6, 3), torch.tensor([1, 0, 1, 0, 1, 0]))


def fuzz_cross_entropy():
    return F.cross_entropy, (torch.randn(4, 5), torch.tensor([0, 3, 1, 2]))
