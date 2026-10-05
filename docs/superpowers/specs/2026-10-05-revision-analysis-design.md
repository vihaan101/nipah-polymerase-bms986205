# Revision computational analysis scripts

Scripts only. The 20 production jobs are written for SunLab and are not launched in this pass. There is one production protocol: 50 ns end to end from the Stage 6 equilibrated structures, with the same Langevin seeds. The 10 ns step count, tier label, default paths, and `10ns` script names are removed from the production and eval scripts. On-disk 10 ns trajectories are left where they are; the scripts no longer target them.

Out of scope: section H (AM1-BCC vs RESP) and the optional 9VXV MD / single-point MM-GBSA.

## How this will be built

Design choices already settled stay fixed: 50 ns end to end from Stage 6, two ligands on 9VXV, SunLab queues, no optional charge or 9VXV MD work.

1. This file is the spec. Do not reopen settled questions. Do not commit unless asked.
2. Test-driven development, one behavior at a time. For each slice, write a failing test, run it and confirm the failure, then write the minimum code that passes. No production code before that red run. Update the existing production resume tests so they expect 25,000,000 steps, and rename them off the `10ns` filename.
3. Karpathy constraints on every slice. Change the production runner from 10 ns to 50 ns in place. Do not add a duration switch that keeps a 10 ns mode. One revision entry point only, with the flags listed below. No extra flags, no plugin system, and no handling for failures that cannot happen. Each slice has a check named in the slice list.
4. Code simplifier after each green slice. Simplify only the code just written. Re-run that slice’s tests and confirm behavior is unchanged.

Slice order, each with its own failing test first:

- Production script is 50 ns only: `TOTAL_STEPS` is 25,000,000, and a search of the production and eval scripts finds no `10ns` or `5_000_000` production length.
- MD queue schedules 20 fake tasks, skips complete sentinels, and respects the parallel cap.
- CPU and GPU waves assign slots and cap ProLIF jobs.
- Pocket RMSD series, block-average sign check, replicate spread, and entropy extreme-frame share on synthetic CSVs.
- H875 / W806 / W730 occupancy rejects a topology missing those residues.
- Frozen-scale composite projection and the weight-sweep rank change.
- Deliverable tables list the required figure groups, and the methods note contains the sweep grid and 0.1 ns spacing.

## One runner

All revision work is `Stages/revision/run_revision.py`. Flags select the step. `--parallel N` overrides the default concurrency. Combining flags runs those steps in order.

- `--production` — 50 ns MD queue. Calls the renamed Stage 7 production script once per replicate.
- `--eval` — existing `md_eval` metrics, CPU wave then GPU wave.
- `--diagnostics` — per-replicate pocket RMSD, MM-GBSA blocks and spread, entropy ΔE, and H875 / W806 / W730 occupancy.
- `--9vxv` — two-ligand pose check and frozen-scale composite comparison.
- `--sweep` — composite-score sensitivity on the existing ranked CSV.
- `--deliverables` — tables, figures, and the methods note.
- `--all` — `--eval`, `--diagnostics`, `--9vxv`, `--sweep`, then `--deliverables`. It does not start MD.

Example: `python Stages/revision/run_revision.py --all --parallel 4`

Functions stay in that file. Tests import them from `run_revision`. Do not add `run_revision_eval.py`, `run_9vxv_two_ligand_check.py`, `sweep_composite_score.py`, `make_revision_deliverables.py`, or `write_methods_note.py`.

## SunLab parallelism

Follow the Stage 6 pattern in `Stages/Stage 6/run_stage6.py`: independent subprocesses, NVIDIA MPS when more than one job shares a GPU, one log file per job, and a batch that keeps going when one job fails.

`--production` and `--eval` auto-detect GPUs with `nvidia-smi`. `--parallel` overrides:

- **MD queue.** 20 independent tasks (4 cases × 5 replicates). Default concurrency is `4 × GPU count`, matching the Stage 6 H100 note (4-way MPS per GPU). Each worker gets `CUDA_VISIBLE_DEVICES` for its assigned GPU. Finished replicates (`TASK_COMPLETE` with `final_step` at 25,000,000) are skipped so a rerun only fills gaps. Failed pairs are listed for `--resume`.
- **Analysis queue.** Split work into a CPU wave and a GPU wave so MM-GBSA and decomp do not fight RMSD/PLIF for the device.
  - CPU wave: backbone/pocket RMSD, ligand RMSD, RMSF, PCA, H-bonds, contacts, PLIF, pocket volume, plus the new revision summaries. Up to one process per case at a time. Cap ProLIF `STAGE7_PLIF_N_JOBS` at `cpu_count / n_cases` so four cases do not each spawn 10 threads.
  - GPU wave: MM-GBSA and per-residue decomp. Use `spawn` (already required in decomp for CUDA). Default **2 GPU slots per device** under MPS, not 20, because each energy worker holds an OpenMM context. Replicates of a case become the parallel unit inside that cap. If a worker dies on an out-of-memory error, the orchestrator retries that replicate alone.
- Composite-score sweep and figure export stay single-process; they only rescore a CSV. 9VXV Vina for two ligands runs its seeds concurrently, same idea as `Stages/Stage 1/run_redocking.py`.

## A. 50 ns production, end to end

Edit `Stages/Stage 7/run_production_10ns_direct_v2.py` in place, then rename it to `run_production_50ns_direct.py`. Do not leave a second 10 ns entry point.

- `TOTAL_STEPS = 25_000_000` (50 ns at 2 fs). Tier label `50ns_direct`. Same 1 ns chunk (`CHUNK_SIZE = 500_000`), checkpoint, 4 ps DCD interval (`DCD_INTERVAL = 2000`), barostat, and seed `sha256("{case}_rep{N}")`.
- Default results root for both the production script and `md_eval` is the repo-level `stage7_50ns_direct/results/<case>/replicate_<N>/`. The revision runner passes that root explicitly.
- Bootstrap only from Stage 6 `equilibrated_state_xml`. Do not append old 10 ns trajectories.
- Update `Stages/Stage 7/tests/test_stage7_10ns_direct_resume.py` and `Stages/Stage 7/tests/test_stage7_10ns_direct_verifier.py` to the new module name, tier, and step count, and rename those test files off `10ns`.
- In `Stages/common/md_eval/stage7_eval_common.py`, the `md_eval` analyzers, `Stages/Stage 7/analysis_10ns_direct/`, `Stages/Stage 8/run_eval.sh`, and `Stages/Stage 8/run_full_eval_10ns_direct_sunlab.sh`, replace campaign identifiers that say `10ns` (`stage7_10ns_direct`, `analysis_10ns_direct`, `*_10ns_direct`, `RMSD_10NS_DIR`, docstrings that call the run a 10 ns production). Use `50ns_direct` names, and rename files whose names contain `10ns`. Leave saved result files on disk alone.
- `--production` on `run_revision.py` is the MD queue. `--resume` restarts an interrupted replicate from its own checkpoint by passing `--resume` through to the production script.

Trajectory validation for this tier expects 12,500 frames and a last frame at 50,000 ps (`TIER_TIME_RANGES["50ns_direct"] = (4.0, 50000.0)`).

## B–E. Re-analysis on the 50 ns trajectories

`--eval` sets `STAGE7_DIRECT_RESULTS_ROOT` and `STAGE7_EVAL_WORK_DIR=Stages/revision/results/eval`, then runs the existing `md_eval` modules through the CPU/GPU waves. That reuses per-replicate MM-GBSA CSVs, PLIF, contacts, decomp, and interaction entropy.

`--diagnostics` adds what those modules do not report. All four cases, five replicates, no unlabeled pooling.

- **Pocket RMSD, per replicate.** Same 10 Å pocket backbone selection as `analyze_backbone_rmsd.py` (`backbone and (around 10.0 resname UNK)`). One series per replicate, plus early-vs-late drift via `split_window_drift`.
- **MM-GBSA time series, per replicate.** Totals from `*_mmgbsa_repN.csv` (`time_ns`, `dG_bind_kcal_mol`). Overlay decomp time series for the highlighted residues when those CSVs exist.
- **Block and running averages.** 1 ns blocks via `block_average`. For each WT vs W730A pair (`A_ERDRP_WT`/`B_ERDRP_MUT`, `C_BMS_WT`/`D_BMS_MUT`) and each ligand pair (`A_ERDRP_WT`/`C_BMS_WT`, `B_ERDRP_MUT`/`D_BMS_MUT`), record whether the sign of the difference is the same in both halves of production.
- **Replicate spread.** Mean, SD, and min/max across five replicates for every reported MM-GBSA endpoint, with `C_BMS_WT` (BMS-986205/WT) called out. Compare each WT–mutant and ligand–ligand gap to the pooled replicate SD. Summaries stay descriptive, not affinity claims.
- **Interaction entropy.** Per replicate: SD of ΔE before log-sum-exp, the ΔE time series, and the share of the log-sum-exp from the most extreme frame (softmax weight of the largest β(ΔE − mean ΔE) term). Primary estimate uses **0.1 ns spacing** (MM-GBSA stride 25): about 500 frames over 50 ns. Also report a 100-frame even subsample of that same 50 ns trajectory.
- **Contacts.** Occupancy versus time for **H875, W806, and W730** (expected resnames HIS/TRP/ALA), per replicate, from the PLIF frame table or a heavy-atom cutoff. Fail if those residue IDs are absent.

CPU revision summaries run in the CPU wave after the metric CSVs exist. They are independent per case and use a process pool.

## F. 9VXV, two ligands only

`--9vxv` does this. No library redock and no MD. Vina seeds for the four complexes (2 ligands × WT/W730A) run concurrently. The ligands are ERDRP-0519 and BMS-986205 only.

- Clean 9VXV the same way Stage 1 cleans 9KNZ, then PDBQT.
- Map 9KNZ Trp730 onto 9VXV by sequence alignment and require TRP before PyMOL mutagenesis to ALA.
- Dock the existing ERDRP-0519 and BMS-986205 PDBQTs. Box center from the co-crystallized ERDRP site; box size matched to the 9KNZ config.
- Binding-mode comparison: superimpose pocket Cα, then heavy-atom ligand RMSD against the 9KNZ pose.
- Report raw composite ingredients side by side, and project both ligands onto the frozen mean and SD in `Stages/Stage 3/results/library_100_mutation_ranked.csv`. Do not re-rank the 100-compound library. State which pose and score differences are small versus structure-dependent.

Frozen statistics are the mean and population SD of the five composite ingredients already used by `add_mutation_composite_scores`: `wt_affinity`, `mut_affinity`, `max(delta_affinity, 0)`, `abs(delta_dist)`, and `pose_center_shift`. The projected score is the weighted inverse-z sum minus the ghost penalty. The library table is not rewritten.

## G. Composite-score sensitivity

`--sweep` rescores the existing ranked CSV only. The three grids below are evaluated separately, not as one cross of every axis at once.

A priori grid, written into the methods note:

- Five descriptor weights, each in `{0, 0.5, 1, 2}` (baseline is all 1s). Full factorial at the current ghost cutoffs (2.5 Å / 3.2 Å) and the current penalties (severe 2, moderate 0.5).
- Ghost penalties at those current cutoffs: severe in `{0, 1, 2, 4}`, moderate in `{0, 0.5, 1}`, weights held at 1.
- Threshold check at baseline penalties: severe cutoff in `{2.0, 2.5, 3.0}` Å, moderate cutoff in `{3.0, 3.2, 3.5}` Å, weights held at 1.

Outputs: BMS-986205 rank under every setting, Spearman correlation versus the published `rank_mutation_aware` order, and whether Δaffinity stays positive for compounds ranked above BMS-986205.

## I. Deliverables

`--deliverables` writes tables and figures under `Stages/revision/results/deliverables/` for convergence, replicate spread, entropy diagnostics, the 9VXV comparison, and the sensitivity sweep. The same flag writes the methods note: 50 ns length, seed rule, SunLab concurrency defaults, entropy spacing, 9VXV prep, and the sweep grid. It does not edit the manuscript.

## Tests

Tests live in `Stages/revision/tests/` and are written before the matching production code. They use synthetic CSVs and fake tasks. They do not start OpenMM, Vina, or a GPU. The renamed production resume tests expect 25,000,000 steps and the `50ns_direct` tier.

## Self-review

Settled choices are unchanged: one 50 ns protocol, no duration switch, one runner, two 9VXV ligands, no RESP, no 9VXV MD, no 100-compound redock, jobs not launched.

Resolved wording, not new scope:

- The production script’s default results root is the repo-level `stage7_50ns_direct/results`, the same root `md_eval` uses. The previous script-relative default under `Stages/Stage 7/` is not kept.
- The sweep is three separate grids (weights, penalties, cutoffs). A single cross of all three axes is not required.
- `Stages/common/md_eval/stage7_production_validation.py` keeps the older continuation-tier keys (`2ns`, `10ns`, `20ns`, `50ns`) because those rows describe other protocols. The direct campaign entry becomes `50ns_direct` at 50,000 ps. The campaign-string search covers the production script, `md_eval` analyzers, `stage7_eval_common.py`, the verifier, the renamed analysis directory, and the Stage 8 shells. It does not treat that historical tier table, `Stages/README.md`, or saved result directories as campaign scripts.
- Env name `STAGE7_EVAL_10NS_DIR` becomes `STAGE7_EVAL_50NS_DIR`. Metric output directory names in code become `*_50ns_direct`. Existing CSV trees on disk are not renamed.
