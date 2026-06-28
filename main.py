from __future__ import annotations

import argparse

import numpy as np

from preprocessing.feature_extraction import extract_all_features
from preprocessing.feature_merge import merge_feature_tables
from preprocessing.feature_normalization import fit_transform_standard
from src.agn_augmentation import augment_with_agn
from src.metrics import binary_metrics
from src.soft_voting import fit_soft_voting
from src.utils import print_metrics, read_fasta, save_predictions
from src.xgb_model import fit_xgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="iTP-AGN-XGB: therapeutic peptide identification with multi-feature representation and AGN."
    )
    parser.add_argument("--train", required=True, help="training FASTA path")
    parser.add_argument("--test", required=True, help="test FASTA path")
    parser.add_argument("--use_agn", action="store_true", help="enable additive Gaussian noise augmentation")
    parser.add_argument("--sigma", type=float, default=0.01, help="Gaussian noise standard deviation")
    parser.add_argument("--n_aug", type=int, default=9, help="augmented samples generated per training sample")
    parser.add_argument("--model", choices=["xgb", "soft_voting"], default="xgb", help="model type")
    parser.add_argument("--n_models", type=int, default=5, help="number of soft-voting base learners")
    parser.add_argument("--output", default="predictions.csv", help="prediction CSV output path")
    return parser.parse_args()


def build_features(fasta_path: str):
    records = read_fasta(fasta_path)
    features = merge_feature_tables(extract_all_features(records))
    return records, features


def main() -> None:
    args = parse_args()

    train_records, train_features = build_features(args.train)
    test_records, test_features = build_features(args.test)

    X_train = train_features.drop(columns=["id", "label"]).to_numpy(dtype=float)
    y_train = train_features["label"].to_numpy(dtype=int)
    X_test = test_features.drop(columns=["id", "label"]).to_numpy(dtype=float)
    y_test = test_features["label"].to_numpy(dtype=int)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_test, _ = fit_transform_standard(X_train, X_test)

    if args.use_agn:
        X_train, y_train = augment_with_agn(
            X_train,
            y_train,
            sigma=args.sigma,
            n_aug=args.n_aug,
            random_state=42,
            include_original=True,
        )

    if args.model == "xgb":
        model = fit_xgb(X_train, y_train, random_state=42)
    else:
        model = fit_soft_voting(X_train, y_train, n_models=args.n_models, random_state=42)

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)

    metrics = binary_metrics(y_test, y_pred, y_score)
    print_metrics(metrics)
    save_predictions(args.output, test_records["id"].tolist(), y_test, y_pred, y_score)
    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()

