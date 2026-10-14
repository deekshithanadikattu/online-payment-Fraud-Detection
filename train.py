"""
Online Payment Fraud Detection: training script (v2).

Fixes over model1.py
- Uses the full dataset instead of the first 20,000 rows.
- Time-based split on `step` (train on earlier hours, test on later ones), so the
  model is evaluated the way it would be used: on future transactions.
- Same feature formulas for train and test (the old script flipped the sign on test).
- Drops account IDs (nameOrig, nameDest): they can't generalise to new accounts.
- Drops isFlaggedFraud: it is the old rule-based system's output, not an input.
- Handles class imbalance with scale_pos_weight.
- Reports imbalance-aware metrics (PR-AUC, precision, recall, F1) instead of accuracy.
- Chooses the decision threshold by minimising business cost on a validation period,
  then reports results on a held-out test period the threshold never saw.

Usage:
    python train.py                      # expects data.csv in this folder
    python train.py --data path/to/data.csv --review-cost 25
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

FEATURES = [
    "hour_of_day",
    "is_transfer",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_balance_change",
    "destination_balance_change",
    "error_balance_orig",
    "error_balance_dest",
    "empties_origin",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering. Used identically by train.py and app.py."""
    out = pd.DataFrame(index=df.index)
    out["hour_of_day"] = df["step"] % 24
    out["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        out[c] = df[c]
    # How much each account's balance moved (same formula everywhere).
    out["origin_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    out["destination_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]
    # Gap between what the balances say and what the amount says should have happened.
    out["error_balance_orig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    out["error_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    # Transaction drains the origin account completely.
    out["empties_origin"] = (
        (df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)
    ).astype(int)
    return out[FEATURES]


def business_cost(y_true, y_prob, amounts, threshold, review_cost):
    """Missed fraud costs the transaction amount; each false alarm costs a manual review."""
    flagged = y_prob >= threshold
    missed_fraud_cost = amounts[(y_true == 1) & ~flagged].sum()
    false_alarm_cost = review_cost * ((y_true == 0) & flagged).sum()
    return missed_fraud_cost + false_alarm_cost


def pick_threshold(y_true, y_prob, amounts, review_cost):
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), [0.5]]))
    costs = [business_cost(y_true, y_prob, amounts, t, review_cost) for t in grid]
    return round(float(grid[int(np.argmin(costs))]), 2)


def evaluate(name, y_true, y_prob, amounts, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fraud_dollars = amounts[y_true == 1].sum()
    caught_dollars = amounts[(y_true == 1) & (y_pred == 1)].sum()
    m = {
        "rows": int(len(y_true)),
        "fraud_cases": int(y_true.sum()),
        "fraud_rate_pct": round(100 * y_true.mean(), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "threshold": threshold,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "accuracy": round((tp + tn) / len(y_true), 6),
        "always_legit_accuracy": round(1 - y_true.mean(), 6),
        "pct_fraud_dollars_caught": round(100 * caught_dollars / fraud_dollars, 2) if fraud_dollars else None,
        "pct_transactions_flagged": round(100 * y_pred.mean(), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    print(f"\n=== {name} ===")
    for k, v in m.items():
        print(f"  {k:28s} {v}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--review-cost", type=float, default=25.0,
                    help="Assumed cost of manually reviewing one flagged transaction (same currency as amount).")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Features to leave out (robustness check), e.g. --exclude error_balance_orig error_balance_dest")
    ap.add_argument("--no-save", action="store_true",
                    help="Print results only; don't overwrite the saved model.")
    args = ap.parse_args()
    unknown = set(args.exclude) - set(FEATURES)
    if unknown:
        raise SystemExit(f"Unknown feature(s) in --exclude: {sorted(unknown)}")
    feats = [f for f in FEATURES if f not in args.exclude]
    if args.exclude:
        print(f"Robustness check: excluding {args.exclude}")

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} rows, {int(df['isFraud'].sum()):,} fraud cases "
          f"({100 * df['isFraud'].mean():.3f}%).")

    # ---- Quick EDA: where does fraud happen? ----
    by_type = df.groupby("type")["isFraud"].agg(["count", "sum", "mean"])
    by_type.columns = ["transactions", "fraud", "fraud_rate"]
    print("\nFraud by transaction type:\n", by_type.to_string())
    fraud_types = by_type.index[by_type["fraud"] > 0].tolist()
    print(f"\nModelling only types with any fraud: {fraud_types}. "
          f"Other types are treated as low risk by rule.")
    df = df[df["type"].isin(fraud_types)].copy()

    # ---- Time-based split: 60% / 20% / 20% of the timeline ----
    q60, q80 = df["step"].quantile([0.6, 0.8])
    train = df[df["step"] <= q60]
    val = df[(df["step"] > q60) & (df["step"] <= q80)]
    test = df[df["step"] > q80]
    print(f"\nSplit by step: train<= {q60:.0f} ({len(train):,}), "
          f"val ({len(val):,}), test> {q80:.0f} ({len(test):,})")

    X_tr, y_tr = add_features(train)[feats], train["isFraud"].values
    X_va, y_va = add_features(val)[feats], val["isFraud"].values
    X_te, y_te = add_features(test)[feats], test["isFraud"].values

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        early_stopping_rounds=30,
        tree_method="hist",   # much faster on large data
        max_bin=128,
        n_jobs=-1,
        verbosity=0,
    )
    print("\nTraining (progress every 25 trees; stops early when validation stops improving)...", flush=True)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=25)
    print(f"Trained {model.best_iteration + 1} trees (scale_pos_weight={spw:.1f}).")

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    amt_va, amt_te = val["amount"].values, test["amount"].values

    threshold = pick_threshold(y_va, p_va, amt_va, args.review_cost)
    print(f"\nCost-optimal threshold on validation (review cost = {args.review_cost}): {threshold:.2f}")

    val_m = evaluate("Validation (threshold chosen here)", y_va, p_va, amt_va, threshold)
    test_m = evaluate("TEST (held-out future period): report these", y_te, p_te, amt_te, threshold)
    test_default = evaluate("Test at default 0.5 threshold (for comparison)", y_te, p_te, amt_te, 0.5)

    cost_opt = business_cost(y_te, p_te, amt_te, threshold, args.review_cost)
    cost_default = business_cost(y_te, p_te, amt_te, 0.5, args.review_cost)
    cost_none = amt_te[y_te == 1].sum()
    print(f"\nTest-period cost: no model {cost_none:,.0f} | threshold 0.5 {cost_default:,.0f} "
          f"| tuned {threshold:.2f} {cost_opt:,.0f}")

    importance = dict(sorted(
        zip(feats, model.feature_importances_.round(4).tolist()),
        key=lambda kv: -kv[1]))
    print("\nFeature importance:", importance)

    if args.no_save:
        print("\n--no-save: model files left unchanged.")
        return
    model.save_model("fraud_model.json")
    meta = {
        "features": feats,
        "modelled_types": fraud_types,
        "all_types": by_type.index.tolist(),
        "threshold": threshold,
        "review_cost": args.review_cost,
        "fraud_by_type": by_type.reset_index().to_dict(orient="records"),
        "split_steps": {"train_max": float(q60), "val_max": float(q80)},
        "metrics": {"validation": val_m, "test": test_m, "test_at_0.5": test_default},
        "test_cost": {"no_model": float(cost_none), "threshold_0.5": float(cost_default),
                      "tuned": float(cost_opt)},
        "feature_importance": importance,
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved fraud_model.json and model_meta.json")


if __name__ == "__main__":
    main()
