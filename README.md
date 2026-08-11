# VTaC Alarm Threshold Transportability

Code accompanying a manuscript prepared for submission to the *Journal of Medical Systems*.

## Question

At a validation-derived high-sensitivity threshold, how many false ventricular tachycardia alarms can be suppressed, and does the unchanged threshold retain its operating-point utility in held-out and external data?

## Data

Raw waveform data are not redistributed.

- VTaC 1.0: PhysioNet, DOI 10.13026/8td2-g363, CC BY-SA 4.0. Obtain directly from PhysioNet.
- PhysioNet/CinC Challenge 2015 v1.0.0: PhysioNet, DOI 10.13026/c9fg-a467, ODC Attribution License v1.0. Obtain directly from PhysioNet.

Expected local layout after download:

```text
02_data/raw/vtac-1.0/
02_data/raw/challenge-2015-1.0.0/
```

## Reproduction modes

Full reproduction: verify releases, run release-native preprocessing, TRAIN-only grouped CV, train five fixed-seed models, lock thresholds from validation, then apply frozen thresholds to test and external cohorts. Commands are implemented in `src/` and require raw data obtained by the user.

Analysis-only reproduction: use `results/aggregate/` with `src/evaluation/vtac_safety_thresholds.py` and `scripts/stage2_consolidate.py` to reproduce aggregate tables and figures. Individual prediction scores and model weights are not released.

## Safeguards

Validation is not used for model selection. Test and external data are not used for model development, calibration, channel mapping, seed selection, or threshold selection. The realtime window is [alarm-2500, alarm) at 250 Hz; no post-alarm samples enter model input. The VTaC release required a deterministic release-native reconstruction; details are in `docs/`.

The five weights are not released. They can be recreated from official VTaC TRAIN with the frozen configuration in `configs/`; model hashes are retained in the provenance manifests.

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
pytest -q
```

## License and attribution

The MIT license applies to code in this repository. Third-party datasets remain subject to their original licenses and are not redistributed. Some workflow code is adapted from the MIT-licensed ML-Health/VTaC repository; see `CODE_PROVENANCE.csv`.

## Citation

See `CITATION.cff`. Cite the VTaC and Challenge 2015 dataset publications separately when using their data.
