# Public Release Security Audit

Audit scope: `public_release_staging/vtac-alarm-threshold-transportability`.

- Raw waveform files: absent.
- Model checkpoints and prediction scores: absent.
- Credentials, tokens, cookies, `.env`, SSH keys: absent.
- Absolute local paths and user-directory references: absent from public files.
- Personal/private manuscript scratchpads: absent.
- Candidate release files: code, tests, configurations, aggregate results, figures, non-sensitive documentation, and non-sensitive provenance only.

Two internal historical audit scripts that emit local physical paths were intentionally excluded. The release also excludes all internal `04_analysis/runs`, raw release manifests containing local physical paths, and model weights.

Status: PASS for staged files, subject to a final tracked-file scan before any remote push.
