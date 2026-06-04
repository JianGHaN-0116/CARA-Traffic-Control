# Data Directory

This directory contains the preprocessed window archives used by the experiments.

## Required Data Files

Due to size constraints, the full data files are not included in this artifact.
Please download the original datasets and run the preprocessing pipeline.

### Datasets

1. **Edge-IIoTset**: https://github.com/ICL-ml4csec/Edge-IIoTset
2. **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html

### Preprocessing

After downloading the datasets, run the preprocessing pipeline:

```bash
python src/preprocessing/build_windows.py --dataset edge_iiotset
python src/preprocessing/build_windows.py --dataset cicids2017
```

### Expected Files

After preprocessing, this directory should contain:

```
data/
├── processed/
│   ├── edge_iiotset/
│   │   ├── train_windows.pkl
│   │   ├── val_windows.pkl
│   │   ├── test_windows.pkl
│   │   ├── scaler.pkl
│   │   └── feature_meta.yaml
│   ├── edge_iiotset_chrono/
│   │   ├── train_windows.pkl
│   │   ├── val_windows.pkl
│   │   └── test_windows.pkl
│   └── cicids2017_cap10000_ws25_thr07/
│       ├── train_windows.pkl
│       ├── val_windows.pkl
│       └── test_windows.pkl
└── raw/
    └── (original dataset files)
```

## Using Pre-computed Data

If you have access to the pre-computed data, place it in the `processed/` subdirectory.
