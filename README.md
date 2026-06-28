# iTP-AGN-XGB

English | [简体中文](README.zh.md)

## Overview

Therapeutic peptide sequence identification using multi-feature representation, additive Gaussian noise (AGN) augmentation, and XGBoost.

This repository provides a compact Python implementation for therapeutic peptide identification. The pipeline includes FASTA parsing, eight handcrafted feature groups, feature merging, standardization, optional additive Gaussian noise augmentation, and XGBoost-based classification.

## Features

- FASTA input with labels in headers, for example `seq_001|1`.
- Eight handcrafted feature groups: AAC, DPC, ASDC, AAIndex, PAAC, APAAC, CTD, and BLOSUM62.
- Feature merging and standardization.
- Optional AGN training-set augmentation.
- Single XGBoost classifier or soft-voting XGBoost ensemble.
- Binary metrics: ACC, SEN/Recall, SPE, PRE, F1, MCC, ROC-AUC, and PR-AUC.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --train Exampledata/train.fa --test Exampledata/test.fa --use_agn --model xgb
python main.py --train Exampledata/train.fa --test Exampledata/test.fa --use_agn --model soft_voting --n_models 3
```

Prediction results are saved to `predictions.csv` by default. Use `--output` to specify another path.

## FASTA Format

Each record title must contain a sequence ID and binary label:

```text
>seq_001|1
KLLKLLKKLLKLLK
>seq_002|0
AGVTSLAGTAA
```

`1` denotes a therapeutic peptide and `0` denotes a non-therapeutic peptide.

## Command-Line Arguments

- `--train`: training FASTA path.
- `--test`: test FASTA path.
- `--use_agn`: enable additive Gaussian noise augmentation.
- `--sigma`: Gaussian noise standard deviation, default `0.01`.
- `--n_aug`: number of augmented samples per training sample, default `9`.
- `--model`: `xgb` or `soft_voting`.
- `--n_models`: number of base learners for soft voting, default `5`.
- `--output`: prediction CSV path, default `predictions.csv`.
