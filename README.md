# MLP-warm-started Bayesian optimization for corner-aware LDO sizing (SKY130 / GF180MCU)

Code, run logs, trained models and data sets for the paper

> S. V. Rajmohan, M. Bharanitharan and M. Gowtaam, "Warm-Starting Bayesian Optimization with a Learned MLP
> Surrogate for Process-Corner-Aware LDO Sizing on Open-Source SKY130 and GF180MCU PDKs," under review, 2026.

The flow sizes an external-capacitor LDO (PMOS pass array, 9-transistor wide-swing cascode error amplifier, PMOS
source-follower buffer) with Gaussian-process Bayesian optimization in ngspice. Every candidate is simulated at five
process corners against eight specifications. A multilayer perceptron trained on earlier simulations proposes BO's
starting designs and is fine-tuned online from the run's own evaluations.

Main result (GF180MCU, 250 simulations, ten seeds): at 3.3 V / 4 µF the warm start met every specification within
three simulations on all seeds, against 1–3 of 10 seeds for cold BO, differential evolution and random search. A prior
trained only on ~5k logged evaluations of earlier BO runs works as well as the full 60.8k-design data set. At an unseen
output voltage (5 V) the warm start is worse than random starts; a fallback rule detects this. Details in the paper.

## Layout

| Path | What it is |
| --- | --- |
| `ldo_pdk.py` | PDK device tables (SKY130, GF180MCU), netlist generation, corner sets |
| `ldo_opt.py` | Objective: corner simulations, metrics, penalty function |
| `ldo_flow.py` | Staged sizing flow (pass-device slicing, BO, refinement, re-verification, `--warm mlp`) |
| `surrogate_ws.py` | Warm-start MLP: data loading, training, proposals, online retraining |
| `gen_dataset.py` | Data-set generation (Sobol, local perturbations, BO designs, test split) |
| `train_surrogate.py`, `active_sample.py` | Analysis MLP, learning curve and active-sampling round |
| `overnight_batch.py` | Original equal-budget comparison (random, DE, cold BO, warm BO) and full-flow runs |
| `revision_batch.py` | Revision experiments: 10 seeds, 15 µF, stored-design baseline, runs-only prior, fallback/ensemble, tree baselines |
| `verify_designs.py` | Temperature (−40/27/125 °C) and Monte Carlo mismatch verification |
| `make_figures.py`, `sky130_compare.py` | Figures and the SKY130 cold/warm comparison |
| `ldo_gui.py` | Small Flask front end for the flow (not used in the evaluation) |
| `runs/` | Logged single runs (each `evals.jsonl` holds every simulated design and its metrics) |
| `runs_batch/` | Equal-budget experiment: one folder per spec / method / seed, hold-out models, summaries |
| `runs_sky130_compare/`, `verify_out/`, `designs_frozen/` | SKY130 runs, verification results, frozen designs |
| `surrogate_ws/`, `surrogate/` | Trained warm-start MLPs and the analysis MLP's metrics |
| `dataset/` | GF180MCU data sets as `.jsonl.xz` (or on the Releases page, see below) |

## Setup (Ubuntu or WSL2 Ubuntu)

1. ngspice 45.2 and Python 3.12 or newer (tested with 3.14.4).
2. PDKs through [Volare](https://github.com/efabless/volare), pinned to `0fe599b2afb6708d281543108caf8310912f54af`,
   installed so that these exist: `~/pdks/gf180mcuD/libs.tech/ngspice` and `~/pdks/sky130A/libs.tech/ngspice`.
   For SKY130 also run `python make_sky130_trim.py` once (builds a 5-V-device-only model library, faster to load).
3. The code expects to live in `~/ldo_project/optimizer`:
   ```bash
   git clone https://github.com/<user>/<repo>.git ~/ldo_project/optimizer
   cd ~/ldo_project/optimizer
   python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
   ```
   For a CPU-only PyTorch: `./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu`.
4. Put `set num_threads=1` in `~/.spiceinit`. With several parallel workers, ngspice's default multi-threading
   slows runs down drastically.
5. Data sets: `./get_data.sh`, then `export LDO_DATASET=~/ldo_project/optimizer/dataset`.
   The warm start itself does not need them (see the runs-only result), only data-set-based training does.

## Reproducing the paper

```bash
# one sizing run (GF180, 3.3 V, 4 uF), cold or warm-started
./venv/bin/python ldo_flow.py --pdk gf180 --vout 3.3 --cout 4
./venv/bin/python ldo_flow.py --pdk gf180 --vout 3.3 --cout 4 --warm mlp --surrogate surrogate_ws/gf180.pt

# tables from the logged results already in runs_batch/ (no simulation)
./venv/bin/python revision_batch.py report

# rerun the revision experiments (resumable; about 6-7 h on a 12-thread desktop)
./venv/bin/python revision_batch.py plan
./venv/bin/python revision_batch.py queue --parts core,extrap,lookup,runsonly --jobs 4 --workers 3
./venv/bin/python revision_batch.py ens       # fallback / ensemble decisions, no SPICE
./venv/bin/python revision_batch.py screen    # random-forest / gradient-boosting baselines, no SPICE
```

| Paper item | Source |
| --- | --- |
| Table III (surrogate R²) | `surrogate_ws/*.pt.json`, `runs_batch/models/holdout_*.pt.json` |
| Table IV (screening) | `runs_batch/screen_baselines.json` |
| Table V, Fig. 3 (equal-budget comparison) | `runs_batch/per_seed.json`, `runs_batch/report_rev.txt` |
| Table VI (proposal accuracy) | `runs_batch/gf180_*/bo_warm_s*/warm_seeds.json` with `evals.jsonl` |
| Fallback and ensemble | `runs_batch/ens_decisions.json` |
| Table VII (full flow) | `runs_batch/flow_summary.json`, `runs_sky130_compare/summary.json` |
| Tables VIII–IX (designs, verification) | `designs_frozen/`, `verify_out/` |
| Fig. 4 (learning curve) | `surrogate/metrics.json` |

Note: the `bo_warm_sobol*` folders in `runs_batch/gf180_3.3V_6.8uF/` are an ablation that turned out to be invalid
(all Sobol designs are at 4 µF, so those priors have no information about C_OUT). They are kept for transparency and
are not used in the paper.

## Data format

Each line of a data-set or `evals.jsonl` file is one corner-evaluated design: the eight sizes `x`
(`Win, Lin, Wtail, n_pass, Wfol, Wbn2, Wmirror, Wvb`), `vout`, `cout_uF`, `source`, `split`, and per-corner metrics
(`tt_line_mV`, `ss_psrr10k_100mA`, `ff_pm_20mA`, …).

## License

Code: MIT (see `LICENSE`). Data sets and run logs: CC BY 4.0.
