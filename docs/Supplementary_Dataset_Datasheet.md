# Supplementary Dataset Datasheet

## VTaC

VTaC v1.0 (PhysioNet DOI 10.13026/8td2-g363) was checksum-verified. It provides labelled VT-alarm events and official TRAIN/validation/test splits. The analysis unit is the alarm event, grouped by waveform record; verified patient identity is unavailable. The release required a deterministic reconstruction because 42 TRAIN alarms had one identifiable raw ECG and raw modality prevalence differed from publication summaries. No event was excluded.

## Challenge 2015

The external cohort was drawn from the official public training partition only: 341 VT records, 89 true and 252 false. It was not used in model development. Challenge signals had source preprocessing before the unchanged frozen VTaC preprocessing, constituting a documented source-processing shift. Historical publications report slightly different Challenge counts; current release labels were retained unchanged.
