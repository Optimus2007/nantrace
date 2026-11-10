import pytest
import torch

from nantrace import NonFiniteError


def make_data(n=64, bad_index=40):
    torch.manual_seed(0)  # also fixes the model weights created after this
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 8, generator=g)
    x[bad_index] = 1e4  # one corrupted sample
    y = torch.randint(0, 4, (n,), generator=g)
    return x, y


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)

    def loss(self, x, y):
        logits = self.fc(x)
        w = logits.exp()  # softmax without the max shift
        p = w / w.sum(-1, keepdim=True)
        return -(p.gather(1, y[:, None]) + 1e-9).log().mean()


def test_lightning_callback_traces_the_bad_batch(tmp_path):
    L = pytest.importorskip("lightning.pytorch")
    from nantrace.integrations.lightning import NantraceCallback

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.head = Head()

        def training_step(self, batch, batch_idx):
            return self.head.loss(*batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    x, y = make_data()
    module = Module()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)
    trainer = L.Trainer(
        max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(tmp_path),
        callbacks=[NantraceCallback()],
    )
    with pytest.raises(NonFiniteError) as err:
        trainer.fit(module, loader)
    r = err.value.report
    assert r.context["step"] == 6  # samples 40..47 are the 6th batch
    assert r.cause_event.op == "aten.exp.default" and "logits.exp()" in r.cause_event.location.code


def test_hf_trainer_traces_the_bad_step(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from transformers import TrainingArguments

    from nantrace.integrations.hf import NantraceTrainer

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()

        def forward(self, x, labels):
            return {"loss": self.head.loss(x, labels)}

    x, y = make_data()
    ds = [{"x": x[i], "labels": y[i]} for i in range(len(x))]
    args = TrainingArguments(
        output_dir=str(tmp_path), per_device_train_batch_size=8, num_train_epochs=1, report_to=[],
        use_cpu=True, save_strategy="no", logging_strategy="no", dataloader_drop_last=False,
        seed=0, data_seed=0,
    )
    torch.manual_seed(0)
    trainer = NantraceTrainer(model=Model(), args=args, train_dataset=ds)
    with pytest.raises(NonFiniteError) as err:
        trainer.train()
    e = err.value.report.cause_event
    # Depending on the weights the corrupted sample overflows exp (inf / inf) or underflows it (0 / 0).
    assert (e.op == "aten.exp.default" and "logits.exp()" in e.location.code) or (
        e.op == "aten.div.Tensor" and "exp underflowed" in e.cause and "test_integrations.py" in e.cause
    )


def test_mcp_server_end_to_end(tmp_path):
    pytest.importorskip("mcp")
    import asyncio
    import json
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    examples = os.path.join(os.path.dirname(__file__), "..", "examples")

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "nantrace", "mcp"])
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert names == {"trace_nan", "fuzz"}
                res = await s.call_tool("trace_nan", {"script": "where_trap.py", "cwd": examples})
                traced = json.loads(res.content[0].text)
                res = await s.call_tool("fuzz", {"path": "fuzz_checks.py", "cwd": examples})
                fuzzed = json.loads(res.content[0].text)
                return traced, fuzzed

    traced, fuzzed = asyncio.run(go())
    assert traced["exit_code"] == 3 and traced["cause"]["grad_fn"].startswith("LogBackward")
    counts = {t["target"]: len(t["findings"]) for t in fuzzed["targets"]}
    assert counts["fuzz_stable_softmax_weights"] == 0 and counts["fuzz_manual_softmax_weights"] == 1
