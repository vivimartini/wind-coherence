"""One AdamW step of AIFS ENS v2, through 100 m wind and a portfolio CRPS. Records memory against rollout length. Stops at the first out-of-memory."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("WN_ROOT", "/rds/user/vm481/hpc-work/wind-coherence"))
REPO = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "aifs-probe"
CKPT = ROOT / "data" / "aifs" / "aifs-ens-crps-2.0.ckpt"
# PROBE_RUNS overrides the list, for example "e:1:1+f:1:4+g:1:7" (name:members:steps).
# The separator is + because sbatch --export splits its value on commas.
RUNS = [
    (name, int(members), int(steps))
    for name, members, steps in (
        item.split(":") for item in os.environ.get("PROBE_RUNS", "a:2:1+b:2:2+c:2:4+d:2:7").split("+")
    )
]
CHUNKS = int(os.environ.get("PROBE_CHUNKS", "16"))
ITERATIONS = 3


def mib(n: int) -> float:
    return round(n / 2**20, 1)


def write(name: str, record: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record, default=str), flush=True)


def farm_nodes(model, torch):
    import numpy as np
    import pandas as pd

    farms = pd.read_csv(REPO / "data" / "interim" / "portfolio.csv")
    coords = model.graph_data["data"].x.detach().cpu().numpy()
    lat, lon = coords[:, 0], coords[:, 1]
    out = []
    for flat, flon in zip(np.deg2rad(farms["lat"]), np.deg2rad(farms["lon"]), strict=True):
        cosd = np.sin(lat) * np.sin(flat) + np.cos(lat) * np.cos(flat) * np.cos(lon - flon)
        out.append(int(np.argmax(cosd)))
    capacity = torch.tensor(farms["capacity_mw"].to_numpy(), dtype=torch.float32)
    return torch.tensor(out), capacity


def stats_for(model, name: str) -> tuple[float, float]:
    di = model.data_indices
    for table in (getattr(di, "name_to_index", None), di.data.output.name_to_index, di.data.input.name_to_index):
        if table is not None and name in table:
            i = table[name]
            return float(model.statistics["mean"][i]), float(model.statistics["stdev"][i])
    raise KeyError(name)


def fair_crps(members, target):
    m = members.shape[0]
    skill = (members - target).abs().mean()
    if m == 1:
        # One member per GPU: the spread term needs the other device's member.
        return skill
    spread = (members[:, None] - members[None, :]).abs().sum() / (2 * m * (m - 1))
    return skill - spread


def one_step(model, inner, optimiser, members: int, steps: int, nodes, capacity, torch):
    di = model.data_indices
    in_prog = di.model.input.prognostic.to("cuda")
    out_prog = di.model.output.prognostic.to("cuda")
    i_u = di.model.output.name_to_index["100u"]
    i_v = di.model.output.name_to_index["100v"]
    (mu_u, sd_u), (mu_v, sd_v) = stats_for(model, "100u"), stats_for(model, "100v")
    grid = model.graph_data["data"].x.shape[0]
    n_in = len(di.model.input.full)

    x = torch.randn(1, model.multi_step, members, grid, n_in, device="cuda")
    target = torch.rand((), device="cuda") * capacity.sum()
    optimiser.zero_grad(set_to_none=True)
    loss = torch.zeros((), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(steps):
            y = inner(x, fcstep=k)
            u = y[0, :, nodes, i_u].float() * sd_u + mu_u
            v = y[0, :, nodes, i_v].float() * sd_v + mu_v
            speed = torch.sqrt(u * u + v * v + 1e-6)
            cf = torch.sigmoid((speed - 8.0) / 1.5)
            portfolio = (cf * capacity).sum(-1)
            loss = loss + fair_crps(portfolio, target) / capacity.sum()
            if k + 1 < steps:
                frame = x[:, -1].clone()
                frame[..., in_prog] = y[..., out_prog].to(frame.dtype)
                x = torch.cat([x[:, 1:], frame.unsqueeze(1)], dim=1)
    loss.backward()
    optimiser.step()
    return float(loss.detach())


def main() -> int:
    import torch

    base = {"model": "ecmwf/aifs-ens-2.0", "gpu": None, "processor_chunks": CHUNKS, "precision": "bf16 autocast"}
    if not torch.cuda.is_available():
        write("probe_error", {**base, "status": "no_cuda"})
        return 1
    base["gpu"] = torch.cuda.get_device_name(0)
    t0 = time.time()
    model = torch.load(CKPT, map_location="cpu", weights_only=False).to("cuda")
    base["load_seconds"] = round(time.time() - t0, 1)
    inner = model.model
    inner.processor.num_chunks = CHUNKS
    inner.processor.chunk_size = inner.processor.num_layers // CHUNKS
    inner.train()
    for p in model.parameters():
        p.requires_grad_(True)
    base["parameters"] = sum(p.numel() for p in model.parameters())
    base["weights_mib"] = mib(torch.cuda.memory_allocated())
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-7)
    nodes, capacity = farm_nodes(model, torch)
    nodes, capacity = nodes.to("cuda"), capacity.to("cuda")
    base["farms"] = int(nodes.numel())

    for name, members, steps in RUNS:
        record = {**base, "run": name, "members": members, "steps": steps, "lead_hours": 6 * steps}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            for _ in range(ITERATIONS):
                torch.cuda.synchronize()
                start = time.time()
                loss = one_step(model, inner, optimiser, members, steps, nodes, capacity, torch)
                torch.cuda.synchronize()
                times.append(round(time.time() - start, 2))
            record.update(
                {
                    "status": "ok",
                    "loss": loss,
                    "seconds_per_step": times[-1],
                    "seconds_all_iterations": times,
                    "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
                    "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
                }
            )
            write(name, record)
        except torch.cuda.OutOfMemoryError as exc:
            record.update(
                {
                    "status": "oom",
                    "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
                    "error": str(exc)[:600],
                }
            )
            write(name, record)
            return 0
        except Exception as exc:
            record.update({"status": "error", "error": f"{exc}\n{traceback.format_exc()}"[-3000:]})
            write(name, record)
            return 1
        optimiser.zero_grad(set_to_none=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
