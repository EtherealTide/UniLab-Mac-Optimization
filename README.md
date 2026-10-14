# UniLab Apple Silicon benchmark

This repository reproduces three UniLab-family MuJoCo recipes on two Apple
Silicon machines, for six runs in total:

| Experiment ID | Recipe | Default iterations |
|---|---|---:|
| `sac_g1_walk_flat` | SAC + MuJoCo + `g1_walk_flat` | 5,000 |
| `flashsac_g1_motion_tracking` | FlashSAC + MuJoCo + `g1_motion_tracking` | 25,000 |
| `flashmimic_g1_motion_tracking` | `g1_flash_mimic/mujoco` task using the FlashSAC learner | 60,000 |

Each recipe runs once on the Mac mini M5 Pro and once on the Mac Studio M5
Ultra. The Studio was initially described as M5 Max, but its collected
`system_profiler` inventory identifies it as M5 Ultra, so the hardware-reported
name is used here. FlashMimic is a motion-imitation task package using FlashSAC,
not a separate learner algorithm.

[REPORT.md](REPORT.md) is generated from the validated result artifacts. Its
only result tables are the six-run table and the three matched-machine
comparisons. The figure plots mean reward against end-to-end wall-clock time.

## Version policy

This protocol does not pin UniLab, unilab-rl, unisim, FlashMimic, Python, or
package versions in this repository. Setup updates the UniLab and FlashMimic
checkouts to their current upstream default branches and lets each checkout
resolve its own dependency configuration. Upstream project constraints and
lockfiles may still constrain what their own environments resolve. The actual
Git revisions, resolved packages, Python version, macOS version, and MPS status
are recorded in `machine-*.json` and each run's `run_config.json`; they are
provenance, not acceptance pins.

UniLab and FlashMimic use separate environments because FlashMimic owns its
task integration and dependency compatibility. No source compatibility patch
is applied by this benchmark repository.

## Deploy on each Mac

Requirements:

- Apple Silicon macOS with Command Line Tools;
- `git`, `curl`, and network access to GitHub, PyPI, and the UniLab asset host;
- at least 20 GiB free storage;
- GitHub authorization for the private `Motphys/FlashMimic` repository.

Install `uv` and clone this repository:

```bash
xcode-select -p || xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

git clone git@github.com:EtherealTide/UniLab-Mac-Optimization.git
cd UniLab-Mac-Optimization
```

Confirm FlashMimic access, then deploy on the Mac mini:

```bash
ssh -T git@github.com
git ls-remote git@github.com:Motphys/FlashMimic.git HEAD

MACHINE_ID=mac-mini-m5-pro \
  ./scripts/setup_macos.sh "$HOME/unilab-mac-benchmark"
```

Use `MACHINE_ID=mac-studio-m5-ultra` on the Studio. The setup script refuses to
update a dirty source checkout, fast-forwards both default branches, creates
the two environments, downloads the G1 assets in both environments,
materializes the motion asset, and records machine/runtime provenance.

For a private/offline FlashMimic transfer, create a bundle on an authorized
machine, copy it to the Mac, and pass it only for the initial clone:

```bash
git -C /path/to/FlashMimic bundle create FlashMimic.bundle --all
scp FlashMimic.bundle user@mac:~/

FLASHMIMIC_BUNDLE="$HOME/FlashMimic.bundle" \
MACHINE_ID=mac-mini-m5-pro \
  ./scripts/setup_macos.sh "$HOME/unilab-mac-benchmark"
```

If a large dependency download is interrupted, rerun setup with longer
transport retries:

```bash
UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
MACHINE_ID=mac-mini-m5-pro \
  ./scripts/setup_macos.sh "$HOME/unilab-mac-benchmark"
```

## Smoke test

Smoke output is separate from formal result data:

```bash
MACHINE_ID=mac-mini-m5-pro \
SMOKE_ROOT="$HOME/unilab-mac-benchmark/smoke/mac-mini-m5-pro" \
  ./scripts/smoke_test.sh
```

The smoke test uses 16 environments and three iterations for each recipe. Use
the analogous machine ID and path on the Studio.

## Run the six training experiments

Keep each Mac on AC power in high-performance mode and close unrelated
compute-heavy applications. On the mini run:

```bash
MACHINE_ID=mac-mini-m5-pro ./scripts/run_unilab_benchmarks.sh
```

On the Studio run:

```bash
MACHINE_ID=mac-studio-m5-ultra ./scripts/run_unilab_benchmarks.sh
```

The three rows execute sequentially and have independent result directories
and live terminal panels. With no override, the runner reads the per-recipe
defaults from [`config/experiments.tsv`](config/experiments.tsv): 5,000,
25,000, and 60,000 iterations. All runs use seed 1, 2,048 environments,
per-iteration TensorBoard logging, MuJoCo, and the MPS learner device selected
by UniLab.

`MAX_ITERATIONS` explicitly replaces the default for every selected row. For a
quick 500-iteration protocol:

```bash
MAX_ITERATIONS=500 \
MACHINE_ID=mac-mini-m5-pro \
  ./scripts/run_unilab_benchmarks.sh
```

To run only one recipe:

```bash
ONLY_EXPERIMENT=flashmimic_g1_motion_tracking \
MACHINE_ID=mac-studio-m5-ultra \
  ./scripts/run_unilab_benchmarks.sh
```

Result directories are never overwritten. Training uses
`training.no_play=true`; video rendering is deliberately separate so playback
does not contaminate the measured training wall time.

## Render the six final policy videos

After all three training runs finish on each machine, render the final
checkpoint from every run:

```bash
MACHINE_ID=mac-mini-m5-pro ./scripts/render_final_videos.sh
```

and on the Studio:

```bash
MACHINE_ID=mac-studio-m5-ultra ./scripts/render_final_videos.sh
```

The script selects the numerically latest `model_<iteration>.pt`, runs policy
playback for 800 steps with one environment, and requires a non-empty
`play_video.mp4`. It writes the MP4, playback log, and start/end metadata into
each experiment directory. Use `VIDEO_STEPS=1200` to change video length,
`ONLY_EXPERIMENT=...` to render one run, and `OVERWRITE_VIDEO=1` to explicitly
replace an existing MP4.

MuJoCo recording on macOS uses GLFW. Run the script from a logged-in macOS
graphical session (a Terminal window is safest); a pure SSH session without a
usable WindowServer may be unable to create frames. The expected six files on
the benchmark hosts are:

```text
results/mac-mini-m5-pro/{sac_g1_walk_flat,flashsac_g1_motion_tracking,flashmimic_g1_motion_tracking}/play_video.mp4
results/mac-studio-m5-ultra/{sac_g1_walk_flat,flashsac_g1_motion_tracking,flashmimic_g1_motion_tracking}/play_video.mp4
```

After collection, the same files live under
`results/raw/<machine>/<experiment>/play_video.mp4` in this repository.

## Collect and analyze

From a workstation clone, collect each machine. The collector includes the
final MP4 files but excludes checkpoints:

```bash
./scripts/collect_results.sh \
  '<mini-user>@<mini-host>' \
  '/Users/<mini-user>/unilab-mac-benchmark/results/mac-mini-m5-pro' \
  results/raw/mac-mini-m5-pro

./scripts/collect_results.sh \
  '<studio-user>@<studio-host>' \
  '/Users/<studio-user>/unilab-mac-benchmark/results/mac-studio-m5-ultra' \
  results/raw/mac-studio-m5-ultra
```

Copy the two `machine-*.json` files from each deployment root into
`results/machines/`, then analyze:

```bash
uv run \
  --with tensorboard==2.21.0 \
  --with matplotlib==3.10.8 \
  scripts/analyze_results.py
```

The analyzer requires the complete 2 × 3 matrix, successful run metadata, and
all six non-empty videos. It reads each run's actual completed iteration count
from `run_summary.json` and checks that value against the run's own benchmark
metadata and resolved training configuration; no separate expected-iteration
argument is needed. TensorBoard scalars are loaded without the library's
default 10,000-event cap, so 25,000- and 60,000-iteration runs are analyzed
in full. A shorter custom run is analyzed using all samples that it contains.

`--allow-missing-videos` is available only for intermediate analysis before
playback artifacts have been collected; a report generated with it is not a
complete submission artifact. Analysis generates:

- `results/processed/curves.csv` with raw reward and performance scalars;
- `results/processed/summary.csv` with validated run summaries;
- `figures/reward_wall_clock.png` with six reward–wall-time curves;
- `REPORT.md` with exactly the two requested result tables.

Wall time for a scalar is its TensorBoard event timestamp minus the wrapper
start time. This includes process startup, environment construction, replay
warm-up, and compilation. The figure applies a ten-event rolling mean only for
display; the CSV retains raw values.

### Downloading the six final videos

Videos are not copied by `collect_results.sh` because checkpoints and MP4s are
large binary artifacts. Use the dedicated collector from the workstation; it
preserves the three experiment directory names and does not overwrite an
existing MP4:

```bash
./scripts/collect_videos.sh \
  'liangyu@192.168.110.52' \
  '/Users/liangyu/unilab-mac-benchmark/results/mac-mini-m5-pro' \
  'results/raw/mac-mini-m5-pro'

./scripts/collect_videos.sh \
  'robot@10.13.1.157' \
  '/Users/robot/unilab-mac-benchmark/results/mac-studio-m5-ultra' \
  'results/raw/mac-studio-m5-ultra'
```

Enter the Mac mini password when `ssh` prompts for it. If a run was rendered
into a different remote results root, replace only the second argument. The
script prints every downloaded MP4 and writes a `VIDEO_SHA256SUMS` file for the
video set. Existing run metadata in the local `results/raw` destination is
preserved.

Steady-state throughput is the time-weighted rate over the latter half of the
run:

```text
tail-half steps/s = sum(environment-step deltas)
                    / sum(Perf/iteration_time)
```

This is not the final iteration's rate and not an unweighted arithmetic mean.
Zero-step iterations retain their elapsed time in the denominator; negative
step deltas are rejected as evidence of mixed or resumed event logs.

## Repository layout

```text
config/experiments.tsv       experiment matrix and default iteration budgets
scripts/setup_macos.sh       latest-source macOS deployment
scripts/run_unilab_benchmarks.sh
scripts/render_final_videos.sh
scripts/collect_results.sh
scripts/analyze_results.py
results/raw/                 TensorBoard logs, metadata, console logs, six MP4s
results/machines/            hardware and resolved runtime provenance
results/processed/           generated curve and summary CSV files
figures/reward_wall_clock.png
REPORT.md                    generated figure plus two core tables
```
