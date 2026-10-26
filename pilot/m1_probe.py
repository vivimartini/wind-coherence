"""One WeatherNextCyclones Mini measurement, in its own process.

A failed or out-of-memory run must not take the rest of the suite with it.
The parent (m1_suite.py) polls the GPU and keeps the record if this process dies.

Official loss path, from docs/weathernext2/wn2_demo.ipynb at weathernext v0.3.0:
one Haiku transform, predictor.loss, jax.value_and_grad. On GPU the mesh
attention implementation is switched to triblockdiag_mha, as that notebook requires.

Rollout rematerialisation uses the released autoregressive wrapper. Its
gradient_checkpointing flag applies hk.remat, which is JAX checkpointing, to
each step inside hk.scan. The released code skips that flag when the target
has a single time step, so the rematerialisation comparison uses 4 and 7 steps
(24 h and 42 h at a 6 h step).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

# Set before JAX is imported. The suite also sets these for the child.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import xarray_jax
from weathernext.utils import checkpoint, data_utils, fiddle_config_io, xarray_tree
from weathernext.weathernext2 import fgn

ROOT = Path(os.environ.get("WN_ROOT", "/rds/user/vm481/hpc-work/wind-coherence"))
PARAMS = ROOT / "data" / "weathernext2" / "WeatherNextCyclones_Mini_<2024.npz"
SAMPLE = ROOT / "data" / "weathernext2" / "source-hres_forecast_init-2024-10-07_res-1.0_levels-13_steps-12.nc"
CONFIG_NAME = "weathernext2/configs/WeatherNextCyclones_Mini"
CHECKPOINT_NAME = "WeatherNextCyclones_Mini_<2024"
CODE_REV = "v0.3.0 89c4b2a77a1c57b328b909c575550fd2e5aadc9c"
WIND_100 = ("100m_u_component_of_wind", "100m_v_component_of_wind")
WIND_10 = ("10m_u_component_of_wind", "10m_v_component_of_wind")
# Mask test, not a training learning rate. Scaled so a nonzero gradient moves
# float32 weights by about 1e-3, which is visible, while untouched weights stay identical.
UPDATE_MAX_STEP = 1e-3


def gpu_static() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc)}
    name, total, uuid, driver = [p.strip() for p in out.split(",")]
    total_mib = float(total)
    return {
        "name": name,
        "memory_total_mib": total_mib,
        "memory_total_gb": round(total_mib / 1024, 2),
        "class": "A100-80GB" if total_mib > 60000 else "A100-40GB" if total_mib > 30000 else "other",
        "uuid": uuid,
        "driver_version": driver,
    }


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wrapper_ctor_name(ctor) -> str:
    name = getattr(ctor, "__name__", None)
    if name:
        return name
    func = getattr(ctor, "func", None)
    if func is not None and getattr(func, "__name__", None):
        return func.__name__
    return type(ctor).__name__


def summarise_wrappers(config) -> list[dict]:
    rows = []
    for i, wrapper in enumerate(config.predictor_wrappers):
        kw = wrapper.get("kwargs", {})
        interesting = {}
        for key in ("num_samples", "gradient_checkpointing", "var_to_clean"):
            if key in kw:
                interesting[key] = kw[key]
        rows.append({"index": i, "constructor": wrapper_ctor_name(wrapper["constructor"]), **interesting})
    return rows


def set_gpu_attention(config) -> dict:
    backend = jax.default_backend()
    ctor = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"]
    keywords = getattr(ctor, "keywords", None)
    if keywords is None:
        raise RuntimeError(f"mesh_model_ctor has no keywords to edit: {type(ctor)}")
    transformer_kwargs = keywords["transformer_kwargs"]
    before = transformer_kwargs.get("attention_type")
    if backend == "gpu":
        transformer_kwargs["attention_type"] = "triblockdiag_mha"
    return {
        "backend": backend,
        "attention_before": before,
        "attention_after": transformer_kwargs.get("attention_type"),
    }


def set_num_samples(config, n: int) -> int:
    found = 0
    for wrapper in config.predictor_wrappers:
        kw = wrapper.get("kwargs", {})
        if "num_samples" in kw:
            kw["num_samples"] = n
            found += 1
    if found != 1:
        raise RuntimeError(f"expected one num_samples wrapper, found {found}")
    return found


def set_gradient_checkpointing(config, enabled: bool) -> bool:
    found = False
    for wrapper in config.predictor_wrappers:
        kw = wrapper.get("kwargs", {})
        if "gradient_checkpointing" in kw:
            kw["gradient_checkpointing"] = enabled
            found = True
    if not found:
        raise RuntimeError("no predictor wrapper exposes gradient_checkpointing")
    return enabled


def wind_report(config, batch: xarray.Dataset) -> dict:
    inputs = list(config.task.input_variables)
    targets = list(config.task.target_variables)
    data_vars = list(batch.data_vars)
    def present(names, pool):
        return {name: name in pool for name in names}
    return {
        "task_inputs": inputs,
        "task_targets": targets,
        "sample_data_vars": data_vars,
        "in_task_inputs": {**present(WIND_100, inputs), **present(WIND_10, inputs)},
        "in_task_targets": {**present(WIND_100, targets), **present(WIND_10, targets)},
        "in_sample_file": {**present(WIND_100, data_vars), **present(WIND_10, data_vars)},
        "hundred_metre_wind_is_a_model_input": all(name in inputs for name in WIND_100),
        "hundred_metre_wind_is_a_model_target": all(name in targets for name in WIND_100),
        "hundred_metre_wind_is_in_the_sample_file": all(name in data_vars for name in WIND_100),
        "ten_metre_wind_is_a_model_input": all(name in inputs for name in WIND_10),
        "ten_metre_wind_is_a_model_target": all(name in targets for name in WIND_10),
    }


def param_module_table(params) -> list[dict]:
    rows = []
    for name, subtree in params.items():
        leaves = jax.tree_util.tree_leaves(subtree)
        rows.append({"module": str(name), "tensors": len(leaves), "parameters": int(sum(x.size for x in leaves))})
    rows.sort(key=lambda r: -r["parameters"])
    return rows


def path_str(path) -> str:
    parts = []
    for item in path:
        key = getattr(item, "key", None)
        if key is not None:
            parts.append(str(key))
        else:
            parts.append(str(getattr(item, "name", item)))
    return "/".join(parts)


def is_trainable(path) -> bool:
    """First mask, chosen after reading the Mini parameter names.

    grid_decoder is the output head. It maps the latent grid state onto the
    predicted atmospheric variables, so an update moves the local forecast,
    including 10 m wind.

    global_norm_conditioning_encoder embeds the stochastic noise input that
    later blocks consume as norm conditioning. Those weights can change both
    the mean and the spread. They are not a dependence-only knob.

    Per-block norm-conditioning linears are left frozen in this first check.
    They are the larger conditioning surface and a later mask, not this one.
    """
    text = path_str(path)
    return ("grid_decoder" in text) or ("global_norm_conditioning_encoder" in text)


def summarise_array(value) -> dict | float | int | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 1 and np.issubdtype(array.dtype, np.number):
        number = float(array)
        return number if np.isfinite(number) else None
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return {"shape": list(array.shape), "dtype": str(array.dtype)}
    return {
        "shape": list(array.shape),
        "mean": None if not np.isfinite(array).any() else float(np.nanmean(array)),
    }


def summarise_diagnostics(diagnostics) -> dict:
    flat: dict = {}

    def rec(prefix, obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                rec(f"{prefix}/{key}" if prefix else str(key), value)
            return
        data = getattr(obj, "data_vars", None)
        if data is not None:
            for key, value in obj.data_vars.items():
                rec(f"{prefix}/{key}" if prefix else str(key), value)
            return
        if hasattr(obj, "shape"):
            flat[prefix or "value"] = summarise_array(obj)

    rec("", diagnostics)
    # Keep the record small. Retain wind variables and the spread we attach.
    kept = {}
    for key, value in flat.items():
        if any(token in key for token in ("10m_", "100m_", "pilot_", "spread")):
            kept[key] = value
    kept["n_diagnostic_entries"] = len(flat)
    return kept


def load_bundle():
    if not PARAMS.is_file():
        raise FileNotFoundError(PARAMS)
    if not SAMPLE.is_file():
        raise FileNotFoundError(SAMPLE)
    with PARAMS.open("rb") as handle:
        ckpt = checkpoint.load(handle, fgn.CheckPoint)
    config = fiddle_config_io.get_fiddle_config_by_name(CONFIG_NAME)
    batch = xarray.load_dataset(SAMPLE).compute()
    return ckpt, config, batch


def extract_batch(config, batch, n_steps: int):
    stop = f"{n_steps * 6}h"
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch,
        target_lead_times=slice("6h", stop),
        **dataclasses.asdict(config.task),
    )
    got = int(targets.sizes.get("time", 0))
    if got != n_steps:
        raise RuntimeError(
            f"asked for {n_steps} target steps ending at {stop}, extraction returned {got}. "
            f"target times: {list(targets.coords['time'].values)}"
        )
    return inputs, targets, forcings


def grad_and_maybe_update(params, grads, do_update: bool) -> dict:
    trainable_elems = 0
    frozen_elems = 0
    trainable_abs = []
    frozen_abs = []
    nonzero_trainable = 0
    trainable_leaves = 0

    def consider(path, grad):
        nonlocal trainable_elems, frozen_elems, nonzero_trainable, trainable_leaves
        array = np.asarray(grad)
        finite = bool(np.isfinite(array).all()) if array.size else True
        max_abs = float(np.nanmax(np.abs(array))) if array.size else 0.0
        record = {"path": path_str(path), "elements": int(array.size), "max_abs": max_abs, "finite": finite}
        if is_trainable(path):
            trainable_elems += array.size
            trainable_leaves += 1
            if max_abs > 0 and finite:
                nonzero_trainable += 1
            if len(trainable_abs) < 12:
                trainable_abs.append(record)
        else:
            frozen_elems += array.size
            if len(frozen_abs) < 8:
                frozen_abs.append(record)

    jax.tree_util.tree_map_with_path(lambda path, g: consider(path, g), grads)
    out = {
        "trainable_rule": "path contains 'grid_decoder' or 'global_norm_conditioning_encoder'",
        "trainable_leaves": trainable_leaves,
        "trainable_elements": trainable_elems,
        "frozen_elements": frozen_elems,
        "trainable_leaves_with_nonzero_finite_grad": nonzero_trainable,
        "trainable_grad_samples": trainable_abs,
        "frozen_grad_samples": frozen_abs,
    }
    if not do_update:
        return out
    if trainable_leaves == 0:
        out["update"] = {"applied": False, "reason": "no parameter path matched the trainable rule"}
        return out
    max_abs = max((row["max_abs"] for row in trainable_abs), default=0.0)
    # The samples above are capped. Compute the true max across the mask.
    true_max = 0.0

    def scan_max(path, grad):
        nonlocal true_max
        if is_trainable(path):
            true_max = max(true_max, float(np.max(np.abs(np.asarray(grad)))))

    jax.tree_util.tree_map_with_path(lambda path, g: scan_max(path, g), grads)
    max_abs = true_max
    if max_abs == 0 or not np.isfinite(max_abs):
        out["update"] = {"applied": False, "reason": "trainable gradient is zero or non-finite", "max_abs_grad": max_abs}
        return out
    lr = UPDATE_MAX_STEP / max_abs

    def apply(path, param, grad):
        if is_trainable(path):
            return param - jnp.asarray(lr, dtype=param.dtype) * grad
        return param

    updated = jax.tree_util.tree_map_with_path(apply, params, grads)
    changed = 0
    unchanged_trainable = 0
    frozen_moved = 0
    checked = 0

    def compare(path, before, after):
        nonlocal changed, unchanged_trainable, frozen_moved, checked
        checked += 1
        delta = np.max(np.abs(np.asarray(after) - np.asarray(before)))
        moved = bool(delta > 0)
        if is_trainable(path):
            if moved:
                changed += 1
            else:
                unchanged_trainable += 1
        elif moved:
            frozen_moved += 1

    jax.tree_util.tree_map_with_path(compare, params, updated)
    out["update"] = {
        "applied": True,
        "learning_rate_for_mask_test": lr,
        "intended_max_step": UPDATE_MAX_STEP,
        "leaves_checked": checked,
        "trainable_leaves_changed": changed,
        "trainable_leaves_unchanged": unchanged_trainable,
        "frozen_leaves_changed": frozen_moved,
        "mask_respected": frozen_moved == 0 and changed > 0,
    }
    return out


def build_fns(config, params):
    @hk.transform
    def loss_fn(inputs, targets, forcings):
        predictor = fgn.construct_predictor(config)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    def grads_fn(params, inputs, targets, forcings):
        def aux(params, inputs, targets, forcings):
            loss, diagnostics = loss_fn.apply(params, jax.random.PRNGKey(0), inputs, targets, forcings)
            return loss, diagnostics

        (loss, diagnostics), grads = jax.value_and_grad(aux, has_aux=True)(params, inputs, targets, forcings)
        return loss, diagnostics, grads

    return jax.jit(grads_fn)


def run(args) -> dict:
    t_load = time.perf_counter()
    ckpt, config, batch = load_bundle()
    attention = set_gpu_attention(config)
    if args.members is not None:
        set_num_samples(config, args.members)
    checkpointing = None
    if args.checkpointing is not None:
        checkpointing = set_gradient_checkpointing(config, bool(args.checkpointing))
    wind = wind_report(config, batch)
    modules = param_module_table(ckpt.params)
    record_time = [str(v) for v in batch.coords["time"].values] if "time" in batch.coords else []
    record = {
        "mode": args.mode,
        "checkpoint_name": CHECKPOINT_NAME,
        "checkpoint_identity_note": (
            "Weights file is weathernext2/params/WeatherNextCyclones_Mini_<2024.npz, "
            "the name in the WeatherNext v0.3.0 README. The description stored inside "
            "the checkpoint is recorded separately and is the string the file itself carries."
        ),
        "checkpoint_description": ckpt.description,
        "checkpoint_license": ckpt.license,
        "training_cutoff": (
            "WeatherNextCyclones_Mini_<2024 is trained on data through 2023 "
            "(WeatherNext README, repository v0.3.0). That cutoff precedes every "
            "evaluation window in DESIGN.md. The earliest listed window, v1, starts 2025-07-02."
        ),
        "code_revision": CODE_REV,
        "config_name": CONFIG_NAME,
        "params_path": str(PARAMS),
        "params_sha256": sha256(PARAMS),
        "sample_path": str(SAMPLE),
        "sample_sha256": sha256(SAMPLE),
        "host": platform.node(),
        "gpu": gpu_static(),
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "attention": attention,
        "wrappers": summarise_wrappers(config),
        "gradient_checkpointing": checkpointing,
        "members": args.members,
        "target_steps": args.steps,
        "wind": wind,
        "sample_sizes": {k: int(v) for k, v in batch.sizes.items()},
        "sample_times": record_time,
        "parameter_modules": modules,
        "parameter_count": int(sum(row["parameters"] for row in modules)),
        "load_seconds": round(time.perf_counter() - t_load, 3),
        "crps_note": (
            "The released FGN loss uses the unbiased CRPS estimator, which divides by "
            "n*(n-1). It is undefined at one member. One member is still profiled for memory."
        ),
    }
    if args.mode == "inventory":
        record["status"] = "ok"
        return record

    inputs, targets, forcings = extract_batch(config, batch, args.steps)
    record["batch"] = {
        "inputs": dict(inputs.sizes),
        "targets": dict(targets.sizes),
        "forcings": dict(forcings.sizes),
        "input_vars": list(inputs.data_vars),
        "target_vars": list(targets.data_vars),
    }
    grads_fn = build_fns(config, ckpt.params)
    t0 = time.perf_counter()
    loss, diagnostics, grads = grads_fn(ckpt.params, inputs, targets, forcings)
    jax.tree_util.tree_leaves(grads)[0].block_until_ready()
    first_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    loss2, diagnostics2, grads2 = grads_fn(ckpt.params, inputs, targets, forcings)
    jax.tree_util.tree_leaves(grads2)[0].block_until_ready()
    steady_s = time.perf_counter() - t1
    loss_f = float(np.asarray(loss2))
    record.update(
        {
            "status": "ok",
            "loss": loss_f if np.isfinite(loss_f) else None,
            "loss_finite": bool(np.isfinite(loss_f)),
            "first_call_seconds": round(first_s, 3),
            "steady_call_seconds": round(steady_s, 3),
            "compile_overhead_seconds": round(first_s - steady_s, 3),
            "diagnostics": summarise_diagnostics(diagnostics2),
            "gradients": grad_and_maybe_update(ckpt.params, grads2, do_update=args.update),
        }
    )
    return record


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("inventory", "grad"), required=True)
    p.add_argument("--members", type=int, default=None)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--checkpointing", type=int, choices=(0, 1), default=None)
    p.add_argument("--update", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = run(args)
    except Exception as exc:
        record = {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        args.out.write_text(json.dumps(record, indent=2, default=str))
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}), flush=True)
        return 1
    args.out.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"status": record["status"], "loss_finite": record.get("loss_finite"), "out": str(args.out)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
