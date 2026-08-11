# Supplementary Model Card

## Intended use

Retrospective evaluation of a frozen decision score for suppressing false VT alarms at a validation-derived high-sensitivity threshold. It is not a prospective clinical decision-support system.

## Model and data

Official-style realtime FCN; input is a 4 x 2500 tensor from the final 10 seconds before alarm at 250 Hz, ordered ECG1, ECG2, PPG, ABP. TRAIN comprises 4,060 VTaC alarms from 1,808 waveform records. Epoch 31 was selected in TRAIN-only grouped CV. Five prespecified seed models were retrained on complete TRAIN and averaged.

## Operating thresholds and performance

Primary tau_95 = 0.102635692, selected in validation to maximize FASR subject to point sensitivity >=0.95. Validation/test/external sensitivity was 0.9504/0.9489/0.9551 and FASR was 0.7486/0.6638/0.6270. The score is not an assumed calibrated patient-risk probability.

## Limitations and monitoring

Missing PPG/ABP and a lone ECG are zero-filled deterministically. The model has no prospective deployment evidence and makes no clinical-safety claim. Future deployment would require prospective evaluation, local validation, monitoring of operating-point utility, and governed revalidation before any score/threshold update.
