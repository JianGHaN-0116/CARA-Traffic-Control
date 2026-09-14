# CARA-Traffic-Control

Source code for a calibration-aware detector-assisted traffic-control study on
Edge-IIoT networks.

> **Status: under review.** The manuscript, its supplementary material, the
> result tables, and the pre-computed evidence data are withheld from this
> public repository while the paper is under review. This repository contains
> the code only. Reviewers receive the full artifact through the journal's
> confidential channel.

## What is here

```
configs/            Experiment configuration (datasets, detector, DRL, splits)
src/
  preprocessing/    Dataset cleaning, feature engineering, window construction
  detectors/        Detector training and evaluation (XGBoost / LightGBM / RF / DNN)
  envs/             Traffic-control environment and reward functions
  agents/           DQN and PPO agents
  experiments/      Experiment drivers
  utils/            Metrics, logging, path helpers
ovs_replay/         Open vSwitch / Mininet replay harness (topologies, flow rules,
                    automation scripts)
scripts/            Reproduction drivers
requirements.txt
```

## Setup

```bash
conda create -n cara-tc python=3.10 -y
conda activate cara-tc
pip install -r requirements.txt
```

## Datasets

Not included (size and licensing). Download separately:

- Edge-IIoTset: https://github.com/ICL-ml4csec/Edge-IIoTset
- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html

Place them under `data/raw/` and run the preprocessing pipeline in
`src/preprocessing/`.

## OVS/Mininet replay

Requires root and Open vSwitch. See `ovs_replay/README.md`. The replay harness
builds a multi-switch topology, installs flow rules derived from controller
actions, and collects traffic and operational metrics.

## Citation

```bibtex
@article{jiang2026calibration,
  title={Calibration-Aware Evaluation of Detector-Assisted Traffic Control in Edge-IIoT Networks},
  author={Jiang, Han and Peng, Lizhi and Tian, Jianying and Jia, Zhongtian and Mu, Qi and Lu, Shengcai},
  journal={Internet of Things},
  note={Under review},
  year={2026}
}
```

## License

Code: MIT License.
