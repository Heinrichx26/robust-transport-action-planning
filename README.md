# Robust Transportation Action Planning

This repository contains the reproducibility materials for the manuscript **"From Predicted Transportation Loss to Robust Action Plans under Limited Capacity."**

The study asks when predicted transportation loss provides a sound priority for a limited action plan. It links each target to a physical action, evaluates that action under a shared set of structural transportation models, and solves for the largest complete-plan value under the least favorable declared setting. The applications cover pre-departure airline coordination, five-minute freeway control, and pre-hour shared-mobility rebalancing.

## Contents

- `src/methodology/`: structural response models, robust plan optimization, sequential plan calibration, result summaries, and figures.
- `src/ccerts/`: data preparation, pre-decision features, prediction models, adapted comparison mechanisms, and constrained score selection.
- `results/structural_robust_results/`: stored numerical results for all rolling folds and the cross-domain summaries used by the manuscript.

Source transportation records remain with their public providers:

- United States Bureau of Transportation Statistics Airline On-Time Performance data;
- METR-LA freeway-speed data for Los Angeles;
- Divvy public trip data for Chicago.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Data preparation

Each preparation program lists its source and output arguments through `--help`.

```bash
python src/ccerts/prepare_bts.py --help
python src/ccerts/prepare_road.py --help
python src/ccerts/prepare_divvy.py --help
```

Place source records under `data/open/bts`, `data/open/road`, and `data/open/divvy`, or pass another location through the documented arguments.

## Smoke tests

Check the grouped robust-rounding construction on a small instance:

~~~bash
python src/methodology/test_matroid_rounding_smoke.py
~~~

The check solves a fractional robust plan, decomposes it into feasible integer
plans, verifies every total and area limit, and compares sampled plan values
with the enumerated integer optimum.

Run one ordered fold with at most 2,500 records per period before a full analysis:

```bash
python src/methodology/run_structural_robust.py \
  --dataset bts \
  --input data/open/bts/ccerts_bts_ready_v2.csv \
  --output-dir results/structural_robust_smoke \
  --smoke
```

Repeat with `--dataset road` and `--dataset divvy` and their prepared input files. A valid smoke test reports zero negative responses, zero responses above service loss, feasible plans, and zero reported mixed-integer solver gap.

The replay inputs can also be checked without fitting a response model:

```bash
python src/methodology/validate_structural_replay.py \
  --dataset bts \
  --input data/open/bts/ccerts_bts_ready_v2.csv \
  --output results/replay_validation/bts.csv
```

Repeat the command for the roadway and shared-mobility inputs. The output records response boundaries and domain-specific physical input checks.

## Full rolling analyses

```bash
python src/methodology/run_structural_robust.py --dataset bts \
  --input data/open/bts/ccerts_bts_ready_v2.csv \
  --output-dir results/structural_robust_results

python src/methodology/run_structural_robust.py --dataset road \
  --input data/open/road/ccerts_road_ready_v3.csv \
  --output-dir results/structural_robust_results

python src/methodology/run_structural_robust.py --dataset divvy \
  --input data/open/divvy/ccerts_divvy_ready.csv \
  --output-dir results/structural_robust_results
```

The program keeps model fitting, mechanism tuning, response-representation selection, plan calibration, and testing in calendar order. Every decision rule receives the same action limit and area limit.

The result tables include the paired improvement over the feasible loss-priority plan, the number of changed actions, and the supported per-action model-discrepancy tolerance. The summary program calculates a capacity-level release rule from earlier paired blocks and applies it to the later evaluation period.

## Summaries and figures

Sequential coverage can be recomputed directly from stored plan results:

```bash
python src/methodology/recompute_adaptive_coverage.py \
  --root results/structural_robust_results

python src/methodology/summarize_final.py \
  --root results/structural_robust_results

python src/methodology/plot_final.py \
  --root results/structural_robust_results/combined \
  --output-dir figures
```

The plotting command reads stored tables and performs no model fitting.

## Reproducibility notes

All random seeds, action capacities, parameter grids, structural settings, area limits, calibration targets, and solver tolerances are set in the source files. The stored tables include plan-level results, parameter choices, feasibility diagnostics, paired tests, and sequential coverage.
