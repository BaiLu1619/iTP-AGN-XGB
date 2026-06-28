from __future__ import annotations

from sklearn.ensemble import VotingClassifier

from src.xgb_model import build_xgb_model


def build_soft_voting_model(n_models: int = 5, random_state: int = 42) -> VotingClassifier:
    if n_models < 1:
        raise ValueError("n_models must be at least 1")

    estimators = []
    for i in range(n_models):
        model = build_xgb_model(random_state=random_state + i)
        model.set_params(
            n_estimators=120 + 30 * i,
            max_depth=3 + (i % 3),
            learning_rate=max(0.02, 0.08 - 0.01 * i),
            subsample=max(0.7, 0.95 - 0.03 * i),
            colsample_bytree=max(0.7, 0.95 - 0.02 * i),
        )
        estimators.append((f"xgb_{i + 1}", model))

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)


def fit_soft_voting(X_train, y_train, n_models: int = 5, random_state: int = 42):
    model = build_soft_voting_model(n_models=n_models, random_state=random_state)
    model.fit(X_train, y_train)
    return model

