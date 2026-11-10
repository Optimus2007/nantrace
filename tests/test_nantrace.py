import math

import pytest
import torch
import torch.nn.functional as F

import nantrace
from nantrace import Guard, NonFiniteError, Tracer


def origin(report):
    assert report.found, report.text()
    return report.origin


def test_exp_overflow_then_div_nan():
    x = torch.tensor([[100.0, 1.0]], requires_grad=True)
    with nantrace.trace() as tr:
        w = torch.exp(x)
        p = w / w.sum(-1, keepdim=True)
    r = tr.explain(p)
    o = origin(r)
    assert o.op == "aten.exp.default" and o.kind == "inf"
    assert "exp overflow" in o.cause and "88.7" in o.cause
    assert o.location.code.startswith("w = torch.exp(x)")
    assert r.chain[-1].op.startswith("aten.div") and "nan" in r.chain[-1].kind


def test_where_trap_backward_only_nan():
    # Forward is finite because where picks 0, but log's gradient at 0 is 0/0.
    x = torch.tensor([0.0, 2.0], requires_grad=True)
    with nantrace.trace() as tr:
        y = torch.where(x > 0, torch.log(x), torch.zeros_like(x)).sum()
        y.backward()
    assert torch.isfinite(y) and torch.isnan(x.grad[0])
    o = origin(tr.explain(x.grad))
    assert o.phase == "backward" and o.grad_fn.startswith("LogBackward")
    assert "torch.log(x)" in o.location.code
    assert "masked" in o.cause and "torch.where(mask, x, 1)" in o.fix


def test_sqrt_at_zero_backward():
    x = torch.zeros(2, requires_grad=True)
    with nantrace.trace() as tr:
        torch.sqrt(x).sum().backward()
    o = origin(tr.explain(x.grad))
    assert o.phase == "backward" and "sqrt" in o.cause
    assert "torch.sqrt(x)" in o.location.code


def test_log_of_sigmoid_underflow():
    x = torch.tensor([-200.0], requires_grad=True)
    with nantrace.trace() as tr:
        loss = -torch.log(torch.sigmoid(x)).sum()
    o = origin(tr.explain(loss))
    assert o.op.startswith("aten.log") and "log(0)" in o.cause
    assert "logsigmoid" in o.fix


def test_fully_masked_softmax_row():
    scores = torch.randn(2, 4)
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    with nantrace.trace() as tr:
        s = scores.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, -1)
    r = tr.explain(a)
    assert r.chain[0].op.startswith("aten.masked_fill")
    last = r.chain[-1]
    assert "softmax" in last.op and "fully masked" in last.cause


def test_float16_overflow_detected_with_exact_value():
    a = torch.full((4, 4), 200.0, dtype=torch.float16)
    b = torch.full((4, 4), 200.0, dtype=torch.float16)
    with nantrace.trace() as tr:
        c = a @ b  # 4 * 200 * 200 = 160000 > 65504
    o = origin(tr.explain(c))
    assert "float16 overflow" in o.cause and "1.6e+05" in o.cause


def test_in_place_op_inputs_are_read_before_mutation():
    x = torch.tensor([1000.0, 1.0])
    with nantrace.trace() as tr:
        x.exp_()
    o = origin(tr.explain(x))
    assert o.inputs[0].max == 1000.0 and "exp overflow" in o.cause


def test_external_nan_input_is_named():
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with nantrace.trace(model) as tr:
        y = model(torch.ones(1, 2))
    o = origin(tr.explain(y))
    assert o.phase == "input" and "weight" in o.cause


def test_module_name_in_report():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with nantrace.trace(model) as tr:
        h = model(torch.ones(1, 2))
        y = torch.log(h - h)
    o = origin(tr.explain(y))
    assert o.module is None or "Sequential" not in (o.module or "")  # log is outside the module
    with nantrace.trace(model) as tr:
        model[0].weight.data.fill_(10.0)
        y = model(torch.full((1, 2), 1e38))
    o = origin(tr.explain(y))
    assert o.module and o.module.startswith("0 (Linear)")


def test_stop_nan_raises():
    with pytest.raises(NonFiniteError) as err:
        with Tracer(stop="nan"):
            torch.tensor([0.0]) / torch.tensor([0.0])
    assert "0 / 0" in str(err.value)


def test_nothing_found():
    with nantrace.trace() as tr:
        torch.ones(3).exp()
    assert not tr.report().found


def test_anomaly_mode_restored():
    before = torch.is_anomaly_enabled()
    with nantrace.trace():
        assert torch.is_anomaly_enabled()
    assert torch.is_anomaly_enabled() == before


# Guard ----------------------------------------------------------------------


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.bn = torch.nn.BatchNorm1d(4)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        logits = self.fc(x) + self.bn(x)
        w = logits.exp()  # unstable on purpose
        return self.drop(w / w.sum(-1, keepdim=True)).pow(2).sum(-1).mean()


def make_batches(bad_step):
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(6):
        x = torch.randn(8, 4, generator=g)
        if i == bad_step:
            x[0] = 1e4
        out.append(x)
    return out


def test_guard_catches_bad_step_and_traces_origin():
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    guard = Guard(net)
    with pytest.raises(NonFiniteError) as err:
        for x in make_batches(bad_step=3):
            guard.step(net, x)
            opt.step()
            opt.zero_grad()
    r = err.value.report
    assert r.context["step"] == 4
    o = origin(r)
    assert o.op == "aten.exp.default" and "logits.exp()" in o.location.code
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_guard_restores_buffers_and_skips():
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    guard = Guard(net, on_error="skip", report=lambda r: None)
    results = []
    for i, x in enumerate(make_batches(bad_step=2)):
        if i == 2:
            mean_before = net.bn.running_mean.clone()
        out = guard.step(net, x)
        results.append(out)
        if i == 2:
            assert torch.equal(net.bn.running_mean, mean_before)
            assert all(p.grad is None for p in net.parameters())
        opt.step()
        opt.zero_grad()
    assert results[2] is None and all(r is not None for i, r in enumerate(results) if i != 2)
    assert len(guard.reports) == 1 and guard.reports[0].found
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_guard_inf_grad_with_finite_loss():
    p = torch.nn.Parameter(torch.zeros(3))
    holder = torch.nn.Module()
    holder.p = p
    guard = Guard(holder)
    with pytest.raises(NonFiniteError) as err:
        guard.step(lambda: torch.sqrt(holder.p).sum())
    r = err.value.report
    assert r.context["loss_finite"] and r.context["bad_grads"] == ["p"]
    assert origin(r).phase == "backward"


def test_guard_is_transparent_when_fine():
    net = torch.nn.Linear(3, 1)
    guard = Guard(net)
    loss = guard.step(lambda: net(torch.ones(2, 3)).pow(2).mean())
    assert math.isfinite(loss.item()) and net.weight.grad is not None


# fuzz -----------------------------------------------------------------------


def test_fuzz_finds_unshifted_softmax_but_not_the_fix():
    v, a = torch.randn(8, 1), torch.randn(8, 1)
    buggy = nantrace.fuzz(lambda v, a: (v.exp() * a).sum(0) / v.exp().sum(0), v, a)
    assert buggy and "exp overflow" in buggy[0].report.cause_event.cause
    assert nantrace.fuzz(lambda v, a: (torch.softmax(v, 0) * a).sum(0), v, a) == []


def test_fuzz_finds_empty_mask_mean():
    f = torch.randn(6, 3)
    labels = torch.tensor([1, 0, 1, 0, 1, 0])
    found = nantrace.fuzz(lambda f, lbl: f[lbl.bool()].mean(0), f, labels)
    assert any("empty" in x.report.cause_event.cause for x in found)


def test_fuzz_finds_gradient_only_bug():
    found = nantrace.fuzz(lambda x: torch.sqrt(x).sum(), torch.rand(4) + 1)
    assert any(x.where == "gradient" for x in found)


def test_fuzz_skips_mutations_that_overflow_the_input():
    # x * 1e4 of a float16 input near its limit would be inf before fn runs
    found = nantrace.fuzz(lambda x: x * 1.0, torch.full((3,), 60000.0, dtype=torch.float16))
    assert found == []


def test_fuzz_reports_bad_example_input():
    found = nantrace.fuzz(lambda x: x.log(), torch.zeros(2))
    assert found and found[0].mutation.startswith("the example input")
