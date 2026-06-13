# Nipah RdRp Inhibitor Resistance MD Workflow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20591160.svg)](https://doi.org/10.5281/zenodo.20591160)
[![DOI — ERDRP-0519 trajectories](https://zenodo.org/badge/DOI/10.5281/zenodo.20591172.svg)](https://doi.org/10.5281/zenodo.20591172)
[![DOI — BMS-986205 trajectories](https://zenodo.org/badge/DOI/10.5281/zenodo.20591174.svg)](https://doi.org/10.5281/zenodo.20591174)
[![DOI — Evaluation outputs](https://zenodo.org/badge/DOI/10.5281/zenodo.20593557.svg)](https://doi.org/10.5281/zenodo.20593557)

Reproducible staged workflow for the Nipah virus RNA-dependent RNA polymerase
(RdRp) inhibitor-resistance molecular modeling study.

This repository is meant to let another researcher inspect, validate, and rerun
the workflow. It contains the stage scripts, configs, tests, compact provenance
outputs, and result summaries. Large generated molecular dynamics system files
are intentionally excluded from Git and listed in `artifacts_manifest.tsv` with
SHA-256 checksums.

Keywords: Nipah virus, NiV, RNA-dependent RNA polymerase, RdRp, antiviral
inhibitor resistance, molecular docking, molecular dynamics, OpenMM, AutoDock
Vina, reproducible computational biology.

## Reproducibility Status

There are two reproducibility modes:

1. **Audit the published snapshot**
   - Uses the tracked source, configs, manifests, CSV/JSON summaries, figures,
     and tests.
   - Does not require rerunning docking or molecular dynamics.
   - Works with a normal clone of this repository.

2. **Rerun the full workflow**
   - Rebuilds receptors/ligands, reruns docking, regenerates Stage 6 systems,
     runs 10 ns OpenMM production trajectories, and runs post-MD analysis.
   - Requires external scientific tools, GPU hardware for Stage 7 production,
     and either regeneration or retrieval of the externalized Stage 6 artifacts.

The first mode is lightweight. The second mode is compute-heavy.

## Repository Layout

```text
Stages/
  Stage 1/    Receptor/ligand preparation and redocking smoke test
  Stage 2/    Historical docking/verification workflow
  Stage 3/    Locked-100 compound selection and resistance screening
  Stage 4/    ADMET, hit verification, and ghost-clash checks
  Stage 5/    Focused five-seed docking validation
  Stage 6/    MD-ready system reconstruction
  Stage 7/    10 ns production MD and post-MD evaluation
  Stage 8/    Standalone/Azure evaluation runner scripts
  common/     Shared docking helper utilities
  data/       Shared receptor, ligand, and reference inputs
```

Important root files:

- `README.md`: this reproducibility guide.
- `requirements.txt`: imported Python packages for the tracked scripts.
- `ARTIFACTS.md`: artifact policy.
- `artifacts_manifest.tsv`: externalized artifact paths, sizes, and checksums.
- `CITATION.cff`: citation metadata for the Zenodo workflow record.
- `LICENSE`: MIT license for code.

## What Is Tracked vs Externalized

Tracked in Git:

- Source scripts for Stages 1 through 8.
- Tests under `Stages/Stage 3/tests`, `Stages/Stage 5/tests`, and
  `Stages/Stage 7/tests`.
- Small configs, manifests, provenance files, CSV/JSON summaries, logs, and
  selected figures.
- Docking inputs/outputs that are small enough to keep the workflow auditable.

Externalized from Git:

- Large Stage 6 generated OpenMM artifacts:
  - `*.chk`
  - `system.xml`
  - `*_state.xml`
  - `topology.pdb`
  - `equilibrated_production_ready.pdb`

These files are listed in `artifacts_manifest.tsv`. The citable workflow record
is archived at Zenodo DOI
[`10.5281/zenodo.20591160`](https://doi.org/10.5281/zenodo.20591160).
Large trajectory and evaluation artifacts are maintained in separate Zenodo
records described in the manuscript data-availability statement.

## Hardware Requirements

Snapshot audit:

- macOS or Linux
- Python 3.10 or 3.11 recommended
- No GPU required

Full rerun:

- Linux strongly recommended for production MD
- NVIDIA CUDA GPU for Stage 7 production MD
- Large local disk or attached storage for trajectories and Stage 6/7 artifacts
- Multiple CPU cores for docking/evaluation

Stage 7 production refuses CPU execution. The code tries CUDA and OpenCL and raises
an error if no GPU platform is available.

## Software Requirements

Recommended base:

- Python 3.10 or 3.11
- Conda or Miniforge/Mambaforge
- Git
- AutoDock Vina
- ADFRsuite 1.0 receptor-preparation tools

Python packages used by the tracked scripts include:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `requests`
- `biopython`
- `rdkit`
- `gemmi`
- `meeko`
- `spyrmsd`
- `MDAnalysis`
- `openmm`
- `openmmforcefields`
- `openff-toolkit`
- `pdbfixer`
- `pytest`

Some of these are much more reliable through conda-forge than pip.

## Environment Setup

Clone the repo:

```bash
git clone https://github.com/vihaan101/nipah-rdrp-inhibitor-resistance-md.git
cd nipah-rdrp-inhibitor-resistance-md
```

Create a conda environment:

```bash
conda create -y -n nipahv-rdrp-md python=3.11
conda activate nipahv-rdrp-md
conda install -y -c conda-forge \
  numpy pandas scipy matplotlib requests biopython rdkit gemmi meeko spyrmsd \
  mdanalysis openmm openmmforcefields openff-toolkit pdbfixer pytest
```

If `openff-toolkit`, `openmmforcefields`, or `pdbfixer` are unavailable in your
platform solver, install them from conda-forge in a fresh Linux/miniforge
environment. This workflow was designed around scientific Python packages that are
often brittle on system Python.

Install PLIF dependency for Stage 7/8 evaluation:

```bash
python -m pip install "prolif==2.1.0"
```

Set a stable Python hash seed for deterministic behavior where Python hashing could
otherwise affect ordering:

```bash
export PYTHONHASHSEED=0
```

## External Tool Setup

### AutoDock Vina

New locked-100 docking scripts accept an explicit Vina path:

```bash
export VINA_PATH=/absolute/path/to/vina
"$VINA_PATH" --version
```

Pass it into Stage 3 and Stage 5:

```bash
python "Stages/Stage 3/run_stage3.py" --locked-100 --vina-path "$VINA_PATH"
python "Stages/Stage 5/run_stage5.py" --focused-100 --vina-path "$VINA_PATH"
```

Historical scripts expect Vina at:

```text
scripts/vina
```

For historical reruns, either create that path or update the relevant script.

### ADFRsuite

Stage 1 expects an ADFRsuite installation path through the `ADFR_ROOT` constant
inside `Stages/Stage 1/prepare_receptors_v2.py`. Use a local path on your machine,
for example:

```text
/opt/ADFRsuite-1.0
```

If rerunning Stage 1, install ADFRsuite 1.0 and either:

1. place it at the path configured in `ADFR_ROOT`, or
2. edit `Stages/Stage 1/prepare_receptors_v2.py` so `ADFR_ROOT` points to your
   local installation.

The script uses:

```text
${ADFR_ROOT}/bin/python
${ADFR_ROOT}/CCSBpckgs/AutoDockTools/Utilities24/prepare_receptor4.py
```

This is not portable yet. It is documented here so reruns fail for an obvious
reason instead of silently diverging.

## Quick Validation After Clone

Run syntax checks:

```bash
python -m compileall -q .
```

Run unit tests:

```bash
python -m pytest \
  "Stages/Stage 3/tests" \
  "Stages/Stage 5/tests" \
  "Stages/Stage 7/tests"
```

Expected result in a complete conda environment: all tests pass.

Observed during repository initialization on local Python 3.13: 29 tests passed and
1 Stage 3 test failed because `rdkit` was not installed in that interpreter. That
is an environment issue, not a known workflow regression.

## Full Reproduction Workflow

Run commands from the repository root unless noted otherwise.

### Stage 1: Receptor and Ligand Preparation

Purpose:

- download PDB/mmCIF entry `9KNZ`
- clean receptor chain
- generate WT and W730A receptor structures
- prepare PDBQT receptor and ligand files
- run a one-compound redocking smoke test

Command:

```bash
python "Stages/Stage 1/prepare_receptors_v2.py"
```

Key inputs:

- downloads `https://files.rcsb.org/download/9KNZ.cif`
- `Stages/Stage 1/config/docking_box.json`
- ADFRsuite receptor preparation tools
- Vina executable expected at `scripts/vina`

Key outputs:

- `Stages/Stage 1/data/9KNZ_clean.pdb`
- `Stages/Stage 1/data/9KNZ_W730A.pdb`
- `Stages/Stage 1/data/9KNZ_clean.pdbqt`
- `Stages/Stage 1/data/9KNZ_W730A.pdbqt`
- `Stages/Stage 1/data/ligands/ERDRP.pdbqt`
- `Stages/Stage 1/results/smoke_test_results.csv`

Reproducibility notes:

- random seed is fixed at `42`
- Vina exhaustiveness is fixed at `16`
- Stage 1 depends on network access to RCSB unless the CIF is already present
- ADFRsuite path is currently hard-coded as described above

### Stage 2: Historical Verification Workflow

Purpose:

- preserve historical docking and verification scripts/results
- provide audit context for the earlier workflow

Typical commands:

```bash
python "Stages/Stage 2/pipeline_v2.py"
python "Stages/Stage 2/verify_v2.py"
python "Stages/Stage 2/reproducibility_script_v2.py"
```

Use Stage 2 for historical audit, not as the primary current reproduction path.
The current locked-100 workflow begins in Stage 3.

### Stage 3: Locked-100 Compound Selection and Screening

Purpose:

- audit compound-selection data
- build deterministic locked-100 Broad-derived library
- dock locked compounds against WT receptor
- dock paired WT/W730A carry-forward set
- produce mutation-aware ranking

Run all locked-100 steps:

```bash
python "Stages/Stage 3/run_stage3.py" \
  --locked-100 \
  --workers 4 \
  --vina-path "$VINA_PATH"
```

Run selected steps:

```bash
python "Stages/Stage 3/run_stage3.py" --locked-100 --step 1
python "Stages/Stage 3/run_stage3.py" --locked-100 --step 2
python "Stages/Stage 3/run_stage3.py" --locked-100 --step 3 --workers 4 --vina-path "$VINA_PATH"
python "Stages/Stage 3/run_stage3.py" --locked-100 --step 4 --workers 4 --vina-path "$VINA_PATH"
```

Underlying scripts:

- `audit_compound_selection_data.py`
- `select_library_100_locked.py`
- `screen_library_100.py --mode wt --resume`
- `screen_library_100.py --mode paired --resume`

Key inputs:

- `Stages/data/lake/Broad_Repurposing_Hub.csv`
- `Stages/data/lake/Broad_Repurposing_Samples.csv`
- Stage 1 receptor PDBQT files
- Vina executable

Key outputs:

- `Stages/Stage 3/data/library_100_locked.csv`
- `Stages/Stage 3/data/library_100_provenance.json`
- `Stages/Stage 3/results/library_100_wt_results.csv`
- `Stages/Stage 3/results/library_100_wt_ranked.csv`
- `Stages/Stage 3/results/library_100_paired_results.csv`
- `Stages/Stage 3/results/library_100_mutation_ranked.csv`
- `Stages/Stage 3/results/library_100_ghost_reconciliation.csv`

Reproducibility notes:

- locked-100 scripts use deterministic provenance files and SHA-256 helpers
- docking steps use `--resume` by default through `run_stage3.py`
- pass an explicit Vina path to avoid relying on local machine layout

### Stage 4: ADMET and Hit Verification

Purpose:

- run ADMET calculations
- verify hit geometry
- check ghost-clash behavior
- write lead-selection/verification summaries

Command:

```bash
python "Stages/Stage 4/run_stage4.py"
```

Individual scripts:

```bash
python "Stages/Stage 4/run_admet_v2.py"
python "Stages/Stage 4/verify_hits_v2.py"
python "Stages/Stage 4/verify_ghost_clash_v2.py"
```

Key outputs:

- `Stages/Stage 4/results/verification/admet_results.csv`
- `Stages/Stage 4/results/verification/admet_results.json`
- `Stages/Stage 4/results/verification/verification_results.csv`
- `Stages/Stage 4/results/verification/ghost_clash_verdict.json`
- `Stages/Stage 4/results/verification/lead_selection.json`
- `Stages/Stage 4/results/verification/PROVENANCE_NOTES.md`

### Stage 5: Focused Five-Seed Validation

Purpose:

- freeze focused validation candidates from Stage 3 mutation-aware results
- run five-seed WT/W730A docking validation
- generate focused seed matrix and validation summaries

Run current focused-100 path:

```bash
python "Stages/Stage 5/run_stage5.py" \
  --focused-100 \
  --workers 4 \
  --vina-path "$VINA_PATH"
```

Run selected focused steps:

```bash
python "Stages/Stage 5/run_stage5.py" --focused-100 --step 1
python "Stages/Stage 5/run_stage5.py" --focused-100 --step 2 --workers 4 --vina-path "$VINA_PATH"
```

Underlying scripts:

- `select_focused_validation_candidates.py`
- `run_focused_five_seed_validation.py --resume`

Key outputs:

- `Stages/Stage 5/results/focused_100_validation_candidates.csv`
- `Stages/Stage 5/results/focused_100_validation_candidates_manifest.json`
- `Stages/Stage 5/results/focused_100_comparator_manifest.json`
- `Stages/Stage 5/results/focused_100_seed_matrix.csv`
- `Stages/Stage 5/results/focused_100_seed_summary.csv`
- `Stages/Stage 5/results/focused_100_seed_failures.csv`

Historical Stage 5 scripts are retained under `Stages/Stage 5/Legacy/`.

### Stage 6: MD-Ready System Reconstruction

Purpose:

- construct four MD-ready systems:
  - `A_ERDRP_WT`
  - `B_ERDRP_MUT`
  - `C_BMS_WT`
  - `D_BMS_MUT`
- write per-case OpenMM systems, equilibrated states, topology, repaired ligand
  files, logs, and manifests

Run all cases sequentially:

```bash
python "Stages/Stage 6/run_stage6.py"
```

Run one case:

```bash
python "Stages/Stage 6/run_stage6.py" --case C_BMS_WT
```

Run four cases concurrently on a CUDA system with NVIDIA MPS:

```bash
python "Stages/Stage 6/run_stage6.py" --parallel 4
```

Key outputs:

- `Stages/Stage 6/results/stage6_manifest.json`
- `Stages/Stage 6/results/<CASE>/_case_manifest.json`
- `Stages/Stage 6/results/<CASE>/ligand_repaired.sdf`
- `Stages/Stage 6/results/<CASE>/_cleaned_for_fixer.pdb`
- externalized artifacts listed in `artifacts_manifest.tsv`

Externalized generated outputs:

- `Stages/Stage 6/results/<CASE>/system.xml`
- `Stages/Stage 6/results/<CASE>/topology.pdb`
- `Stages/Stage 6/results/<CASE>/equilibrated_production_ready.pdb`
- `Stages/Stage 6/results/<CASE>/equilibrated_production_ready_state.xml`
- `Stages/Stage 6/results/<CASE>/equilibrated_production_ready.chk`

To restore the exact published Stage 6 systems, retrieve these files from the
future external archive and place them at the paths listed in
`artifacts_manifest.tsv`. To regenerate them, rerun Stage 6.

### Stage 7: 10 ns Direct Production MD

Purpose:

- run 10 ns production MD directly from Stage 6 equilibrated states
- run five independent replicates per case
- write DCD trajectories, logs, final PDBs, checkpoints, and completion sentinels

Production command for one case/replicate:

```bash
python "Stages/Stage 7/run_production_10ns_direct_v2.py" \
  --case A_ERDRP_WT \
  --replicate 1 \
  --results-root "stage7_10ns_direct/results" \
  --stage6-results "Stages/Stage 6/results" \
  --resume
```

Run all four cases and five replicates with a shell loop:

```bash
for case_id in A_ERDRP_WT B_ERDRP_MUT C_BMS_WT D_BMS_MUT; do
  for rep in 1 2 3 4 5; do
    python "Stages/Stage 7/run_production_10ns_direct_v2.py" \
      --case "$case_id" \
      --replicate "$rep" \
      --results-root "stage7_10ns_direct/results" \
      --stage6-results "Stages/Stage 6/results" \
      --resume
  done
done
```

Stage 7 constants:

- total steps: `5,000,000`
- timestep: `0.002 ps`
- total simulated time: `10 ns`
- chunk size: `500,000` steps
- DCD interval: `2,000` steps
- log interval: `1,000` steps
- temperature: `300 K`
- pressure: `1 bar`
- per-replicate seed: `sha256("{case}_rep{N}") % 2^31`

Key outputs:

- `stage7_10ns_direct/results/<CASE>/replicate_<N>/production.dcd`
- `stage7_10ns_direct/results/<CASE>/replicate_<N>/production.log`
- `stage7_10ns_direct/results/<CASE>/replicate_<N>/production_final.pdb`
- `stage7_10ns_direct/results/<CASE>/replicate_<N>/production.chk`
- `stage7_10ns_direct/results/<CASE>/replicate_<N>/TASK_COMPLETE`
- `stage7_10ns_direct/results/<CASE>/replicate_<N>/_case_manifest.json`

Production outputs are not included in this Git repository. They should be stored
externally because they are trajectory artifacts.

### Stage 7 Evaluation

Purpose:

- verify trajectories are evaluation-ready
- calculate RMSD, RMSF, ligand RMSD, hydrogen bonds, contacts, MM/GBSA,
  decomposition, interaction entropy, PCA, PLIF, and pocket-volume metrics
- generate comparison figures and summary JSON/CSV outputs

Set local environment variables:

```bash
export STAGE7_STAGE6_RESULTS_DIR="$PWD/Stages/Stage 6/results"
export STAGE7_STAGE6_MANIFEST="$PWD/Stages/Stage 6/results/stage6_manifest.json"
export STAGE7_EVAL_10NS_DIR="$PWD/stage7_results_eval_10ns_direct"
export STAGE7_AZURE_ONLY=0
```

Verify trajectories before analysis:

```bash
python "Stages/Stage 7/analysis_10ns_direct/verify_eval_ready_trajectories.py" \
  --results-root "$PWD/stage7_10ns_direct/results" \
  --output-json "$PWD/stage7_results_eval_10ns_direct/trajectory_eval_ready_manifest.json" \
  --output-md "$PWD/stage7_results_eval_10ns_direct/trajectory_eval_ready_summary.md" \
  --fail-on-not-ready
```

Run analysis scripts:

```bash
cd "Stages/Stage 7/analysis_10ns_direct"

python analyze_backbone_rmsd.py
python analyze_ligand_rmsd.py
python analyze_rmsf.py
python analyze_hbond.py
python analyze_contacts.py
python analyze_mmgbsa.py
python analyze_decomp.py
python analyze_interaction_entropy.py
python analyze_pca.py
python analyze_pocket_volume.py
python analyze_plif.py
python make_comparison_figures.py
python make_figure1_docking.py
```

Tracked snapshot outputs are under:

```text
Stages/Stage 7/eval_results_10ns_direct/
```

For a pure rerun, write new outputs to `stage7_results_eval_10ns_direct/` and
compare them against the tracked summaries/figures.

### Stage 8: Standalone/Azure Evaluation Scripts

Stage 8 contains standalone evaluation scripts and Azure-oriented orchestration
wrappers. These are useful when reproducing the evaluation on cloud machines with
mounted Azure Files shares and NVMe staging.

Important scripts:

- `Stages/Stage 8/run_eval.sh`
- `Stages/Stage 8/run_eval_standalone.sh`
- `Stages/Stage 8/run_full_eval_10ns_direct_azure.sh`
- `Stages/Stage 8/run_campaign_10ns_direct_spot.sh`

These scripts expect Azure environment variables such as:

- `MOUNT_POINT`
- `STORAGE_ACCT`
- `STORAGE_KEY`
- `SHARE_NAME`
- `SCRIPTS_SHARE`
- `SHARE_DIR_BASELINE`
- `STAGE7_TARGET_CASE`
- `STAGE7_EVAL_SHARE_DIR`

Use Stage 8 only if reproducing the Azure execution path. For a local rerun, use
the Stage 7 commands above.

## Comparing Rerun Results to the Snapshot

For a clean comparison, keep one checkout as the published snapshot and run the
workflow in a separate checkout or copy. Example:

```bash
export SNAPSHOT_REPO=/path/to/published/nipah-rdrp-inhibitor-resistance-md
export RERUN_REPO=/path/to/rerun/nipah-rdrp-inhibitor-resistance-md

diff -u \
  "$SNAPSHOT_REPO/Stages/Stage 3/data/library_100_provenance.json" \
  "$RERUN_REPO/Stages/Stage 3/data/library_100_provenance.json"

diff -u \
  "$SNAPSHOT_REPO/Stages/Stage 3/results/library_100_mutation_ranked.csv" \
  "$RERUN_REPO/Stages/Stage 3/results/library_100_mutation_ranked.csv"

diff -u \
  "$SNAPSHOT_REPO/Stages/Stage 5/results/focused_100_seed_summary.csv" \
  "$RERUN_REPO/Stages/Stage 5/results/focused_100_seed_summary.csv"
```

Floating-point MD metrics may not be byte-identical across GPU models, driver
versions, OpenMM versions, and platform precision settings. For Stage 7, compare
summary metrics, qualitative conclusions, and success-criteria JSON/CSV outputs,
not raw binary trajectory bytes.

## Known Reproducibility Caveats

- Stage 1 has a hard-coded ADFRsuite path. Fix `ADFR_ROOT` for non-author machines.
- Historical scripts may expect `scripts/vina`; locked-100 scripts support
  `--vina-path`.
- Stage 7 production requires a GPU and refuses CPU execution.
- Some reruns depend on network access, especially the Stage 1 RCSB download.
- Large trajectory and evaluation artifacts are externalized in separate Zenodo
  records from the code/workflow DOI.
- Conda environment solving is platform-sensitive. Linux + conda-forge is the
  recommended baseline for full reproduction.

## Citation

Use `CITATION.cff` for citation metadata. The primary citable workflow record is:

Agrawal V. *Nipah RdRp Inhibitor Resistance MD Workflow*. Zenodo. 2026.
doi:[10.5281/zenodo.20591160](https://doi.org/10.5281/zenodo.20591160).

## License

Code is released under the MIT License. Data files and third-party input datasets
may have their own upstream terms; verify those before redistributing externally.
