# Repository Guidelines

## Project Structure & Module Organization
This repository is organized into two model-specific implementations plus shared paper assets:

- `FlowCache4MAGI-1/`: FlowCache integration for MAGI-1.
- `FlowCache4SkyReels-V2/`: FlowCache integration for SkyReels-V2.
- `assets/`: figures used in the top-level documentation.

Within each model folder, keep code, configs, and scripts separated:

- MAGI-1 code lives under `FlowCache4MAGI-1/inference/`, with run configs in `config/` and `yaml_config/`, and runnable entry scripts in `scripts/`.
- SkyReels-V2 runtime code is under `FlowCache4SkyReels-V2/skyreels_v2_infer/`, with shell entrypoints at the folder root (for example `run_flowcache_fast.sh`).

## Build, Test, and Development Commands
Install dependencies per subproject:

```bash
pip install -r FlowCache4MAGI-1/requirements.txt
pip install -r FlowCache4SkyReels-V2/requirements.txt
```

Run common workflows:

```bash
bash FlowCache4MAGI-1/scripts/single_run/flowcache_t2v.sh
bash FlowCache4MAGI-1/scripts/sample/flowcache_vbench.sh
bash FlowCache4MAGI-1/scripts/metric.sh
bash FlowCache4SkyReels-V2/run_flowcache_fast.sh
bash FlowCache4SkyReels-V2/run_flowcache_kvcompress.sh
```

## Coding Style & Naming Conventions
Use Python with 4-space indentation and PEP 8 naming (`snake_case` for functions/variables, `PascalCase` for classes). Keep module boundaries aligned with existing folders (`pipeline/`, `cache/`, `modules/`, `scheduler/`). Prefer descriptive, lowercase shell script names using underscores (for example `flowcache_v2v.sh`).

## Testing Guidelines
There is no dedicated unit-test suite or coverage gate in this repo today. Validate changes with script-level smoke runs and metric checks:

- generation smoke test via the relevant `bash` runner;
- quality comparison via `FlowCache4MAGI-1/tools/video_metrics.py` (through `scripts/metric.sh`).

For performance changes, record latency/quality deltas and the exact command used.

## Commit & Pull Request Guidelines
Follow the repository’s observed commit style: short, imperative, lowercase subjects (for example `fix kvcache bug`, `update readme`, `add MAGI-1 code`).

PRs should include:

- scope (MAGI-1, SkyReels-V2, or both);
- reproducible run command(s);
- environment notes (GPU, CUDA, key dependency versions);
- before/after results for speed and quality when behavior changes.
