# Artifact Policy

This repository uses a hybrid artifact policy.

Git tracks code, configuration, tests, provenance notes, compact result summaries,
and selected figures. Large generated molecular dynamics systems are excluded from
Git because they make normal clones slow and are better distributed through a
persistent archive or release asset.

`artifacts_manifest.tsv` lists each excluded artifact with:

- relative path from the repository root
- byte size
- SHA-256 checksum
- reason for externalization

The workflow/code record is archived at Zenodo DOI
[`10.5281/zenodo.20591160`](https://doi.org/10.5281/zenodo.20591160). Large
trajectory and evaluation artifacts are maintained in separate Zenodo records:
ERDRP-0519 trajectories at
[`10.5281/zenodo.20591172`](https://doi.org/10.5281/zenodo.20591172),
BMS-986205 trajectories at
[`10.5281/zenodo.20591174`](https://doi.org/10.5281/zenodo.20591174), and
evaluation outputs at
[`10.5281/zenodo.20593557`](https://doi.org/10.5281/zenodo.20593557).

Current external artifact locations: Zenodo records listed above.
