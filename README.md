# UniLab Apple Silicon benchmark

This repository is the auditable test bundle for six UniLab/UniLab-RL MuJoCo
runs on two Apple Silicon systems:

| Experiment | Mac mini | Mac Studio |
|---|---:|---:|
| SAC + MuJoCo + `g1_walk_flat` | M5 Pro | M5 Ultra |
| FlashSAC + MuJoCo + `g1_motion_tracking` | M5 Pro | M5 Ultra |
| FlashMimic + MuJoCo + `g1_flash_mimic` | M5 Pro | M5 Ultra |

The Mac Studio was described as both “M5 Max” and “M5 Ultra” in the original
request. `system_profiler` identifies the tested machine as **Apple M5 Ultra**,
so this repository uses that hardware-reported name. “flashmiic” was interpreted
as the official `Motphys/FlashMimic` task package; FlashMimic uses FlashSAC and
is not a separate learner algorithm.

The measured results, plots, methodology, and limitations are in
[REPORT.md](REPORT.md). Raw TensorBoard logs and run metadata are kept under
`results/raw/`; checkpoints and videos are intentionally excluded.

## Immutable software inputs

The source of truth is [`config/versions.env`](config/versions.env). The
benchmark pins the following immutable inputs:

| Component | Pin used |
|---|---|
| UniLab source for SAC/FlashSAC | `d5aef9f740e081d835738979e9e4e852805b54c3` (`1.3.4`) |
| UniLab release used by FlashMimic lock | `33bb11c4da7df25efdd6c165c9ba23f4de290605` (`1.3.4`) |
| UniLab `uv.lock` SHA-256 | `1a87e4041976fbdb3a1d862db0ef3fa9545c7976f714e4aa93b77c9cf43d1741` |
| unilab-rl | `1.4.11`; release commit `739986e63c623cd93a3c0eb93c7bcfb8f5a01784` |
| unisim-core | `1.7.15`; release commit `07cf9f9e591899cf7855e1b8f36020b6570cf002` |
| FlashMimic | `e0a2b5c10f1b3815de0b830f5c035ae24e95171e` (`0.1.0`) |
| FlashMimic `uv.lock` SHA-256 | `21448a97af54f0df2f6e6282acca9ecebd5c631278afbfae18f2ac28fed2e11e` |
| Python | `3.13.15` |
| PyTorch | `2.14.0` for latest UniLab; `2.14.1` for FlashMimic |
| MuJoCo | `3.11.0` |
| mjbatch-uni | `0.2.5` |

`unilab-rl` and `unisim-core` are installed as immutable versioned wheels by
the upstream frozen lockfiles. Their release commits are informational source
mappings for those wheel releases; deployment enforcement is by exact wheel
version and lockfile checksum. No lockfile is regenerated during setup.

### Why there are two environments

The current UniLab `main` commit removed `TensorMotionCommand`, while the
pinned current FlashMimic source still imports that symbol. Combining those two
heads fails before training. The benchmark therefore uses:

1. the latest pinned UniLab source and its frozen lock for SAC and FlashSAC;
2. the pinned FlashMimic source and its own frozen `uv.lock`, which selects the
   compatible UniLab 1.3.4 release, for FlashMimic.

This preserves upstream code and labels the tested inputs honestly; no local
compatibility patch is applied.

## macOS deployment

### Requirements

- Apple Silicon Mac running macOS 27.0.1 or a compatible newer release;
- Apple Command Line Tools (`xcode-select --install`);
- `git`, `curl`, and network access to GitHub, PyPI, and the UniLab asset host;
- at least 20 GiB free storage;
- GitHub authorization for the private `Motphys/FlashMimic` repository.

The upstream FlashMimic README currently claims Linux/NVIDIA support only.
These runs evaluate its existing Apple MPS code path; they do not expand the
upstream support claim. Its documentation's “CUDA MPS” means NVIDIA
Multi-Process Service, not Apple Metal Performance Shaders. Do not run
`uni-cumps` or set `training.cuda_process_sharing=mps` on macOS.

### 1. Install uv and clone this bundle

```bash
xcode-select -p || xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

git clone git@github.com:EtherealTide/UniLab-Mac-Optimization.git
cd UniLab-Mac-Optimization
```

### 2. Ensure access to FlashMimic

The default setup clones `git@github.com:Motphys/FlashMimic.git`. Confirm that
the logged-in account can read it:

```bash
ssh -T git@github.com
git ls-remote git@github.com:Motphys/FlashMimic.git \
  e0a2b5c10f1b3815de0b830f5c035ae24e95171e
```

For an offline/private transfer, create a bundle on an authorized host and copy
it to the Mac:

```bash
git -C /path/to/FlashMimic bundle create FlashMimic-e0a2b5c.bundle \
  e0a2b5c10f1b3815de0b830f5c035ae24e95171e
scp FlashMimic-e0a2b5c.bundle user@mac:~/
```

Then set `FLASHMIMIC_BUNDLE` when running setup as shown below.

### 3. Create the two frozen environments and prefetch assets

Run this once on each Mac:

```bash
MACHINE_ID=mac-mini-m5-pro \
  ./scripts/setup_macos.sh "$HOME/unilab-mac-benchmark"
```

Use `MACHINE_ID=mac-studio-m5-ultra` on the Studio. With a bundle:

```bash
FLASHMIMIC_BUNDLE="$HOME/FlashMimic-e0a2b5c.bundle" \
MACHINE_ID=mac-mini-m5-pro \
  ./scripts/setup_macos.sh "$HOME/unilab-mac-benchmark"
```

The script:

- checks for Darwin/arm64 and Command Line Tools;
- checks out the exact UniLab and FlashMimic commits in a dedicated directory;
- installs Python 3.13.15 and synchronizes both frozen lockfiles;
- fails closed on a dirty checkout, package mismatch, commit mismatch, or
  unavailable Torch MPS backend;
- prefetches G1 assets in **both** environments and materializes the motion file
  before any timed run;
- records `system_profiler`, package, Git, Python, toolchain, and MPS facts.

Existing development checkouts are not touched.

If a slow PyPI connection resets while downloading a large wheel, rerun the
same frozen setup with longer transport retries; this does not change the lock
or selected artifacts:

```bash
UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
MACHINE_ID=mac-mini-m5-pro \
  ./scripts/setup_macos.sh "$HOME/unilab-mac-benchmark"
```

This retry was needed once on the tested Mac mini for the FlashMimic
environment. The resumed sync still passed both lock checks and all exact
package-version checks.

### 4. Run smoke tests

Smoke outputs are separate from report data:

```bash
MACHINE_ID=mac-mini-m5-pro \
SMOKE_ROOT="$HOME/unilab-mac-benchmark/smoke/mac-mini-m5-pro" \
  ./scripts/smoke_test.sh
```

Use the analogous Studio machine ID and path on that system. The smoke recipe
uses 16 environments and three iterations to exercise asset loading, MuJoCo,
MPS learning, logging, and finalization.

### 5. Run the formal 500-iteration protocol

Keep the Mac connected to AC power, select its high-performance power mode,
close unrelated compute-heavy applications, and run:

```bash
MACHINE_ID=mac-mini-m5-pro ./scripts/run_unilab_benchmarks.sh
```

On the Studio:

```bash
MACHINE_ID=mac-studio-m5-ultra ./scripts/run_unilab_benchmarks.sh
```

The experiment matrix is in [`config/experiments.tsv`](config/experiments.tsv).
All runs use seed 1, the owner configuration's 2048 environments, 500 training
iterations, per-iteration TensorBoard logging, and `training.no_play=true`.
Results are never overwritten. To run one row only:

```bash
ONLY_EXPERIMENT=flashmimic_g1_motion_tracking \
MACHINE_ID=mac-studio-m5-ultra \
  ./scripts/run_unilab_benchmarks.sh
```

## Collecting and analyzing results

From a workstation clone of this repository, collect the allowed artifacts:

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

Copy the `machine-*.json` and `verify-*.json` files from each deployment root
into `results/machines/`. The machine collection script removes stable hardware
serial, UUID, and UDID fields because they are irrelevant to performance and
unsafe to publish. Then render the processed CSV files and single figure:

```bash
uv run \
  --with tensorboard==2.21.0 \
  --with matplotlib==3.10.8 \
  scripts/analyze_results.py
```

The collector refuses to overwrite a non-empty destination and writes a
`SHA256SUMS` manifest after transfer. The plot has three panels
(`Loss/critic`, `Loss/actor`, and `Loss/temperature`). Color identifies the
task/algorithm and line style identifies the machine. Each scalar event uses:

```text
elapsed_seconds = TensorBoard event.wall_time
                  - benchmark_start.json["start_epoch_sec"]
```

This includes process/CLI startup, environment construction, replay warm-up,
and first compilation, not just the interval after the first loss sample. The
wrapper timestamp has one-second resolution. A ten-sample rolling mean is
used only for visualization; `results/processed/curves.csv` retains the raw
scalar values.

## Repository layout

```text
config/                 exact pins and experiment matrix
scripts/                deployment, verification, execution, collection, analysis
results/raw/            canonical TensorBoard logs and run metadata
results/machines/       hardware and runtime inventories
results/processed/      extracted scalar and run summary CSV files
figures/                loss versus wall-clock figure
REPORT.md               protocol, results, interpretation, limitations
```

## Interpretation boundary

The 500-iteration window is a controlled hardware/runtime comparison, not a
claim of policy convergence. SAC, FlashSAC, and FlashMimic do not share an
identical objective or network, so raw loss magnitudes across different rows
must not be read as an algorithm-quality ranking. The report therefore includes
throughput and reward alongside the requested loss curves.
