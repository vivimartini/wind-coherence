# Milestone 1 feasibility

Slurm job 36472459, completed 26 September 2026, 12 minutes 44 seconds, exit 0.
Node class NVIDIA A100-SXM4-80GB. Code `weathernext` v0.3.0, commit `89c4b2a`.
Checkpoint file `WeatherNextCyclones_Mini_<2024`. The description inside the file
is `WeatherNext2_Mini_<2024`. Training data runs through 2023, before every
evaluation window in DESIGN.md.

The official weather loss was differentiated. This is not a power-forecast result.

## What fit

| Run | Result | Device memory | Steady backward | Compile |
| --- | --- | --- | --- | --- |
| 2 members, 1 step | loss 1.26, finite | 66 GB | 3.3 s | 47 s |
| 2 members, 4 steps, rematerialised | loss 1.44, finite | 67 GB | 7.9 s | 50 s |
| 2 members, 7 steps, rematerialised | loss 1.57, finite | 66 GB | 12.1 s | 46 s |

Seven steps is a 42 hour forecast at the model's 6 hour step. Four steps is 24 hours.
The same 4-step and 7-step runs without rematerialisation asked for 528 GB and
873 GB and were killed. Four members at one step asked for 153 GB and was killed.
Eight members was not started.

One member never reached the GPU. The released fair CRPS divides by n(n−1), so it
is undefined for a single member.

## Parameter update

The trainable set was `grid_decoder` and `global_norm_conditioning_encoder`.
Of 376 tensors, 61 were in that set. Sixty of them changed on one masked step.
No frozen tensor changed. 100 m wind is not an input or a target of this
checkpoint. 10 m wind is both.

Raw records: `/rds/user/vm481/hpc-work/wind-coherence/results/m1/`.

## Account limit, read from the login node on 25 October 2026

The ampere association for `mlmi-vm481-sl2-gpu` uses QOS `gpu1`. That QOS allows 64 GPUs per user and a wall time of 1 day 12 hours. It does not cap the account at one GPU. `mybalance` showed 517 GPU hours remaining of 520. A two-GPU job was still required to confirm the scheduler honours that, and was queued as 39193411 with an estimated start of 23:20 UTC the same evening. Section 16 of the plan stays a four-GPU measurement, not a single-GPU story, until those jobs report.

## GPU-hour budget

517 GPU hours remain. On four GPUs that is about 129 hours of wall time. The working allocation, written before the one-GPU spike has reported a steady step time, is:

| Item | GPU hours |
| --- | --- |
| Arms C and D, 3 seeds, 12 wall hours each on 4 GPUs | 288 |
| Arm B | 16 |
| Profiling | 30 |
| Scaling at 1, 2 and 4 GPUs | 15 |
| Sealed evaluation | 20 |
| Total | about 370 |

About 150 hours stay for failed runs and a search whose size is fixed before it is launched. The 12 wall hours per arm is a guess. Replace it with `steps × seconds per step` once `wn-mem-spike` reports the steady step time.
