# From Anomaly Scores to Cost-Sensitive Inspection Decisions

## Data

Copy the `bottle` folder into `data/` (layout in `data/README.md`), or run
`python download_data.py` (needs `pip install huggingface_hub`). To keep the
data somewhere else: `MVTEC_DIR=/path/to/mvtec python run_experiment.py`.

## Run

```
python run_experiment.py
python run_experiment.py --seed-check   # adds a 20-seed Q-learning check
```

The seed is fixed to 42. Package versions are printed at the start of a run.
It takes a few minutes on CPU (most of it is HOG/LBP feature extraction).

## Notes

- Test images are split 50/50 (stratified) into calibration and held-out sets
  for Stage 2. The detector never trains on either half.
- The Stage 1 threshold is the 95th percentile of normal-training scores.
- Table 8 is an in-sample, post-hoc comparison on all 83 test images.
- MVTec AD is licensed CC BY-NC-SA 4.0. Please cite Bergmann et al., CVPR 2019.
