"""Checks every AIFS ENS run must pass before its output is used.

Two failure modes are guarded. A NaN that survives the model's imputer, or
appears in a raw prediction, poisons every later step and every gradient. And
the five constant forcings (`lsm`, `sdor`, `slor`, `wmb`, `z`) are not read
from the checkpoint: `SimpleRunner` warns "Coupled forcings are not supported"
and takes them from the input state, so a missing or zero-filled field would
run silently. They are compared against the checkpoint's own training
statistics instead.
"""

from __future__ import annotations

import numpy as np

STATIC_FIELDS = ("lsm", "sdor", "slor", "wmb", "z")


class NaNGuardError(RuntimeError):
    pass


def install_nan_guard(inner, log: list | None = None):
    """Wrap the encoder-processor-decoder's forward.

    Its input is the tensor after normalisation and imputation, and its output is
    the raw prediction before post-processing; both must be fully finite. The
    post-processed fields may carry NaN again where the input was masked (sea
    points of soil fields, land points of wave fields), which is by design.
    """
    import torch

    original = inner.forward
    calls = {"n": 0}

    def guarded(x, *args, **kwargs):
        calls["n"] += 1
        bad_in = int((~torch.isfinite(x)).sum())
        if bad_in:
            raise NaNGuardError(f"call {calls['n']}: {bad_in} non-finite values in the preprocessed input")
        y = original(x, *args, **kwargs)
        bad_out = int((~torch.isfinite(y)).sum())
        if bad_out:
            raise NaNGuardError(f"call {calls['n']}: {bad_out} non-finite values in the raw output")
        if log is not None:
            log.append({"call": calls["n"], "input_finite": True, "output_finite": True})
        return y

    inner.forward = guarded
    return calls


def check_static_fields(fields: dict, model) -> dict:
    """Each constant forcing must be present, not zero-filled, and match the training statistics."""
    stats = model.statistics
    index = model.data_indices.data.input.name_to_index
    report = {}
    for name in STATIC_FIELDS:
        entry: dict = {"present": name in fields}
        if not entry["present"]:
            entry["ok"] = False
            report[name] = entry
            continue
        values = np.asarray(fields[name], dtype=float)
        finite = np.isfinite(values)
        entry["finite_fraction"] = round(float(finite.mean()), 4)
        entry["nonzero_fraction"] = round(float((values[finite] != 0).mean()), 4) if finite.any() else 0.0
        entry["mean"] = float(np.nanmean(values))
        entry["std"] = float(np.nanstd(values))
        i = index[name]
        entry["train_mean"] = float(stats["mean"][i])
        entry["train_std"] = float(stats["stdev"][i])
        scale = max(entry["train_std"], 1e-9)
        entry["mean_offset_in_train_std"] = round(abs(entry["mean"] - entry["train_mean"]) / scale, 3)
        entry["std_ratio"] = round(entry["std"] / scale, 3)
        entry["identical_frames"] = bool(values.ndim == 2 and np.array_equal(values[0], values[-1], equal_nan=True))
        entry["ok"] = bool(
            entry["finite_fraction"] > 0.2
            and entry["nonzero_fraction"] > 0.01
            and entry["mean_offset_in_train_std"] < 0.5
            and 0.5 < entry["std_ratio"] < 2.0
        )
        report[name] = entry
    return report
