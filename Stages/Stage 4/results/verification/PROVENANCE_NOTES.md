# Stage 4 Verification — Provenance & Quarantine Notes

Date: 2026-06-06
Author: pre-submission audit reconciliation

These notes annotate two files in this directory that conflict with the **final
Stage 5 / Stage 7 evidence** used in the CompBioChem manuscript. They are retained
for pipeline-history transparency but are **superseded** and must **not** be cited
as final evidence. If the primary data package is deposited (journal Research-Data
Option C), either exclude these two files or include this note alongside them.

## 1. `verification_results.csv` — SUPERSEDED docking pass (quarantined)

This file records an **early focused-docking verification pass** for BMS-986205:

| field | value (this file) |
| :--- | :--- |
| wt_affinity | -7.687 |
| mut_affinity | -7.442 |
| delta_affinity | **+0.245** (W730A weakens) |
| ghost_clash (this CSV) | False (ghost_clash_dist 3.269 Å) |

These values **conflict** with the final Stage 5/7 docking signal, which is the
authoritative source for Table 3, Supplementary Table S2, and the main-text claims:

| source | BMS WT | BMS W730A | delta |
| :--- | :--- | :--- | :--- |
| **Final seed-42** (`Stage 7/.../docking_affinity_summary.csv`) | -7.159 | -7.567 | **-0.408** |
| **Final five-seed mean** (same file) | -7.1566 | -7.5420 | **-0.385** |
| This superseded file | -7.687 | -7.442 | +0.245 |

The sign of the W730A effect is **opposite** here (+0.245 vs the final -0.39/-0.41),
because this pass predates the revised receptor preparation, scoring, and five-seed
validation finalized in Stage 5/7. **Do not** use `verification_results.csv` for any
affinity claim.

Note also an internal Stage 4 inconsistency: `ghost_clash_verdict.json` reports
`ghost_clash: true` (ghost_clash_dist 2.149 Å, threshold 2.5 Å), while
`verification_results.csv` reports `ghost_clash: False` (different distance column,
3.269 Å). Both predate the final analysis and are not used in the manuscript.

## 2. `admet_results.csv` — clinical_phase requires reconciliation

`admet_results.csv` records BMS-986205 as `clinical_phase = Phase 3`, with the
override rationale text "compound is in Phase 3 clinical trials." This is a
**pipeline-internal triage label** and conflicts with the manuscript's cited source:

- Manuscript Ref 19 (Luke JJ et al., *Clin Cancer Res* 2025;31(11):2134–2144,
  doi:10.1158/1078-0432.CCR-24-0439) is a **Phase 1/2** study of linrodostat
  (BMS-986205).

The manuscript **prose is already conservative** and does not assert "Phase 3":
it describes BMS-986205 as an "investigational IDO1 inhibitor" with "early-phase
clinical evaluation [19]" (Main Manuscript lines ~25, 167). That hedged wording is
the authoritative description for submission.

Reconciliation options (pick one before depositing this CSV):
1. **Preferred (non-destructive):** keep this note; treat the manuscript's
   "investigational / early-phase" wording as authoritative and flag the CSV's
   `Phase 3` label as a superseded triage tag.
2. Edit `admet_results.csv` `clinical_phase` to the citable evidence
   ("Phase 1/2", per Ref 19) and update the override_rationale text accordingly.
   *(Only do this if you are comfortable editing a historical pipeline output; if
   BMS-986205 reached Phase 3 in another indication, verify and cite that source
   before asserting "Phase 3" in any deposited file.)*

The descriptor values in this CSV (MW 410.92, cLogP 6.58, HBD 1, HBA 2 for
BMS-986205; HBA 7 for ERDRP-0519) match Supplementary Table S3 after the HBA fix and
are not in dispute — only the `clinical_phase` field needs reconciliation.
