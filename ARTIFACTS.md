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

Before a public reproducibility release, upload the listed artifacts to a persistent
location such as Zenodo, institutional storage, or GitHub release assets, then add
the download URL or DOI to this file and `CITATION.cff`.

Current external artifact location: `TBD`
