# Public Release Security Audit

Audit scope: this public repository.

- Raw third-party waveform extensions (`.dat`, `.hea`, `.atr`, `.mat`, `.zip`) and trained checkpoints (`.pt`, `.pth`) are absent.
- `.gitignore` excludes raw data, waveform components, checkpoints, credentials, environments, caches, and private directories.
- Recursive review found no credentials, tokens, API keys, SSH keys, `.env` files, or personal/local-path material in releasable files. Historical internal audit scripts containing local paths were excluded.
- Public result files contain aggregate manuscript-facing values only.
- Model weights are intentionally excluded; they can be recreated from official TRAIN data and frozen configuration.

Status: PASS. The repository was published publicly at https://github.com/wzwmm2006/vtac-alarm-threshold-transportability. Raw third-party data are not redistributed.