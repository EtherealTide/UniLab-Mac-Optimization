# UniLab Apple Silicon performance report

## Executive summary

This report covers six pinned UniLab-family MuJoCo training runs: three
experiment recipes on a Mac mini with Apple M5 Pro and on a Mac Studio with Apple M5 Ultra. All canonical runs completed
with Torch's Apple MPS backend available. Across recipes, the M5 Ultra Studio delivered **2.03–2.54× the tail-half environment-step throughput** and reduced wrapper wall time by **50.0–60.0%**.

![Loss versus wall-clock time](figures/loss_wall_clock.png)

## Tested systems

| System | Model | CPU / GPU | Unified memory | OS | Power |
|---|---|---|---:|---|---|
| Mac mini | Mac17,16, M5 Pro | 18 CPU (12 Performance + 6 Super), 20 GPU | 64 GB | macOS 27.0.1 (26A434) | AC, high performance |
| Mac Studio | Mac17,15, M5 Ultra | 36 CPU, 80 GPU | 96 GB | macOS 27.0.1 (26A434) | AC |

Both systems are arm64, report Metal 4, use Apple Command Line Tools 27 / clang
21, and passed an actual Torch MPS allocation, matrix multiplication, backward,
synchronization, and finite-gradient check. The machine-readable inventories
are in `results/machines/`. The Studio was initially described as an M5 Max,
but `system_profiler` identifies the tested Mac17,15 system as Apple M5 Ultra;
this report uses the measured hardware identity.

## Protocol and pins

Each recipe used seed 1, 2048 environments from its owner configuration, 5,000
iterations, per-iteration TensorBoard logging, and no post-training play. MuJoCo
simulation and manager tensors execute on CPU; the learner uses Apple MPS.

The exact immutable inputs are listed in [`config/versions.env`](config/versions.env)
and summarized in the README. SAC and FlashSAC use UniLab main commit
`d5aef9f740e081d835738979e9e4e852805b54c3`. FlashMimic uses commit
`e0a2b5c10f1b3815de0b830f5c035ae24e95171e` and its own frozen lock, selecting
the compatible UniLab 1.3.4 release. Both environments use unilab-rl 1.4.11,
unisim-core 1.7.15, Python 3.13.15, and MuJoCo 3.11.0.

The separate environment is necessary because current FlashMimic imports
`TensorMotionCommand`, which the latest UniLab main no longer exposes. No
upstream source was patched for this test.

Wall time for every plotted scalar is the TensorBoard event timestamp minus the
wrapper start recorded in `benchmark_start.json`. This includes process startup,
environment creation, replay warm-up, and compilation. G1 robot and motion
assets were materialized before canonical runs; an initial Studio FlashMimic
run that downloaded assets after the timer started was excluded and preserved
separately on the test host.

Steady-state throughput is the total environment-step delta over iterations
2,501–5,000 divided by the sum of their `Perf/iteration_time` values. Each of
these runs therefore contributes 2,500 iterations and 5,120,000 measured
environment steps. This time-weighted aggregate is equivalent to the harmonic
mean of the per-iteration rates because each ordinary iteration advances 2,048
steps. It is not the arithmetic mean and does not use the potentially noisy
final iteration alone.

Three runs contain one zero-step timing sample in the first half: the trainer
recorded an iteration time while the logical environment-step counter did not
advance. The analysis preserves that time in the denominator and counts zero in
the numerator; it rejects only negative step deltas.

## Results

| Machine | Experiment | Status | Iterations | Environment steps | Training wall time (s) | End-to-end wall time (s) | Tail-half throughput (steps/s) | Final mean reward |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mac mini M5 Pro | SAC / G1 walk flat | completed | 5,000 | 10,260,480 | 998.11 | 1,006 | 10,314.05 | 282.7983 |
| Mac Studio M5 Ultra | SAC / G1 walk flat | completed | 5,000 | 10,260,480 | 395.50 | 402 | 26,158.03 | 283.5175 |
| Mac mini M5 Pro | FlashSAC / G1 motion tracking | completed | 5,000 | 10,440,704 | 725.35 | 741 | 13,986.90 | 12.4588 |
| Mac Studio M5 Ultra | FlashSAC / G1 motion tracking | completed | 5,000 | 10,440,704 | 344.06 | 362 | 29,752.11 | 13.8394 |
| Mac mini M5 Pro | FlashMimic task `g1_flash_mimic/mujoco` using FlashSAC | completed | 5,000 | 10,440,704 | 1,432.23 | 1,449 | 7,128.40 | 40.1184 |
| Mac Studio M5 Ultra | FlashMimic task `g1_flash_mimic/mujoco` using FlashSAC | completed | 5,000 | 10,440,704 | 709.53 | 725 | 14,453.95 | 39.8086 |

`Training wall time` is `run_summary.json.training_wall_time_sec`.
`End-to-end wall time` is the benchmark wrapper's integer-second interval from
before CLI process launch through successful process exit.

| Matched recipe | Studio / mini tail-half throughput | Studio training-time reduction | Studio end-to-end reduction |
|---|---:|---:|---:|
| SAC / G1 walk flat | 2.54× | 60.4% | 60.0% |
| FlashSAC / G1 motion tracking | 2.13× | 52.6% | 51.1% |
| FlashMimic task `g1_flash_mimic/mujoco` using FlashSAC | 2.03× | 50.5% | 50.0% |

The requested loss figure contains critic, actor, and temperature loss panels.
Curves use a ten-event rolling mean for readability. The extracted unsmoothed
events are published in `results/processed/curves.csv`.

## Interpretation

The performance comparison should use wall time, environment-step throughput,
and within-recipe curve progression. Loss magnitude itself is not comparable
across SAC, FlashSAC, and the FlashMimic task: their objectives,
observation/action handling, and networks differ. FlashMimic is a G1
whole-body motion-imitation task package whose learner is FlashSAC, rather than
a third standalone learner algorithm. The canonical task owner is
`g1_flash_mimic/mujoco`; its configuration points to the packaged motion file
`dance1_subject2_part.npz`.

Within each matched recipe, the two machines reached similar loss scales after
the same iteration/step budget while the Studio completed sooner. The final
critic loss was 2.642/2.630 (mini/Studio) for SAC, 2.613/2.632 for FlashSAC,
and 1.576/1.582 for the FlashMimic task. These close endpoints are an execution
consistency check, not proof that the learned policies are equivalent.

SAC shows the largest matched-machine advantage: the Studio shortened the
5,000-iteration wrapper interval from 1,006 s to 402 s. Its tail-half throughput
advantage is 2.54×, while training time is reduced by 60.4%. FlashSAC and the
FlashMimic task likewise show approximately twofold throughput and training-time
differences, with the longer window reducing the relative influence of startup
overhead.

Five thousand iterations provide a longer controlled performance window, but do
not establish final policy quality or convergence. Reward is included as
contextual evidence only; these are still single-seed runs.

## Observed warnings and limitations

- FlashMimic's upstream documentation formally targets Linux/NVIDIA. This is an
  Apple MPS evaluation and must not be represented as an upstream support claim.
- FlashSAC emitted a Torch Dynamo recompile-limit warning on both systems.
- CUDA graph capture was skipped for the mixed CPU/MPS execution path, as
  expected.
- A non-writable NumPy array warning and a FlashMimic initial applied-torque
  host-view warning were observed. Neither stopped the completed runs.
- Three runs contained one equal-step, zero-environment-step timing sample in
  the warm-up half; this is retained in the raw event data and included in the
  wall-time denominator.
- The machines used the same frozen dependency graphs but preinstalled uv patch
  releases differed; uv only performed environment synchronization and was not
  in the timed training process.
- Only one seed and one bounded run per experiment-machine pair were requested;
  no confidence interval or run-to-run variance can be reported.

## Audit trail

Every canonical directory contains the TensorBoard event file,
`run_config.json`, `run_summary.json`, the benchmark wrapper's start/end JSON,
and the complete console log. Checkpoints and video are omitted to keep the
submission repository reviewable. Reproduction commands and collection details
are in [`README.md`](README.md). The canonical runs in this report were recorded
between 2026-10-13 05:35 and 08:03 UTC.
