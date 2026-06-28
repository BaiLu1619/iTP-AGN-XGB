from __future__ import annotations

import xgboost as xgb


def build_xgb_model(random_state: int = 42) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        random_state=random_state,
    )


def fit_xgb(X_train, y_train, random_state: int = 42) -> xgb.XGBClassifier:
    model = build_xgb_model(random_state=random_state)
    model.fit(X_train, y_train)
    return model
