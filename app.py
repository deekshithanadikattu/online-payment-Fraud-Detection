"""
Online Payment Fraud Detection: Streamlit app (v2).

Run after train.py:
    streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st
import xgboost as xgb

from train import add_features

FEATURE_LABELS = {
    "hour_of_day": "Hour of day",
    "is_transfer": "Is a TRANSFER",
    "amount": "Amount",
    "oldbalanceOrg": "Origin balance before",
    "newbalanceOrig": "Origin balance after",
    "oldbalanceDest": "Destination balance before",
    "newbalanceDest": "Destination balance after",
    "origin_balance_change": "Origin balance change",
    "destination_balance_change": "Destination balance change",
    "error_balance_orig": "Origin balance mismatch",
    "error_balance_dest": "Destination balance mismatch",
    "empties_origin": "Drains origin account",
}


@st.cache_resource
def load():
    model = xgb.XGBClassifier()
    model.load_model("fraud_model.json")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load()
threshold = meta["threshold"]

st.title("Online Payment Fraud Detection")
st.caption(
    f"XGBoost model trained on earlier transactions, tested on a later period. "
    f"Decision threshold {threshold:.2f}, chosen to minimise missed-fraud losses "
    f"plus review costs."
)

with st.expander("How well does the model do? (held-out test period)"):
    t = meta["metrics"]["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{t['recall']:.1%}")
    c2.metric("Precision", f"{t['precision']:.1%}")
    c3.metric("PR-AUC", f"{t['pr_auc']:.3f}")
    c4.metric("Fraud $ caught", f"{t['pct_fraud_dollars_caught']}%")
    st.write(
        f"Only {t['pct_transactions_flagged']}% of transactions are flagged for review. "
        f"For reference, labelling everything 'legitimate' would already score "
        f"{t['always_legit_accuracy']:.2%} accuracy, which is why accuracy isn't reported."
    )

st.subheader("Transaction details")
col1, col2 = st.columns(2)
with col1:
    types = meta["all_types"]
    tx_type = st.selectbox("Transaction type", types,
                           index=types.index("TRANSFER") if "TRANSFER" in types else 0)
    amount = st.number_input("Amount", min_value=0.0, value=180000.0, step=1000.0)
    step = st.number_input("Time step (hours since start)", min_value=1, value=300)
with col2:
    old_org = st.number_input("Origin balance before", min_value=0.0, value=180000.0, step=1000.0)
    new_org = st.number_input("Origin balance after", min_value=0.0, value=0.0, step=1000.0)
    old_dest = st.number_input("Destination balance before", min_value=0.0, value=0.0, step=1000.0)
    new_dest = st.number_input("Destination balance after", min_value=0.0, value=0.0, step=1000.0)

if st.button("Check transaction", type="primary"):
    if tx_type not in meta["modelled_types"]:
        st.success(
            f"**Low risk.** No {tx_type} transactions were fraudulent in the training data, "
            f"so this type isn't scored by the model."
        )
        st.stop()

    raw = pd.DataFrame([{
        "step": step, "type": tx_type, "amount": amount,
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
    }])
    feats = meta["features"]
    X = add_features(raw)[feats]
    prob = float(model.predict_proba(X)[0, 1])

    if prob >= threshold:
        st.error(f"**Flag for review.** Fraud probability {prob:.1%} (threshold {threshold:.0%}).")
    else:
        st.success(f"**Looks legitimate.** Fraud probability {prob:.1%} (threshold {threshold:.0%}).")

    # Per-feature contributions (SHAP values computed natively by XGBoost).
    contribs = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=feats), pred_contribs=True
    )[0][:-1]
    expl = (
        pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in feats],
                      "value": X.iloc[0].values, "contribution": contribs})
        .assign(abs_c=lambda d: d["contribution"].abs())
        .sort_values("abs_c", ascending=False)
        .head(5)
    )
    st.subheader("Why?")
    st.caption("Top factors. Positive pushes towards fraud, negative towards legitimate (log-odds).")
    for _, r in expl.iterrows():
        direction = "raises" if r["contribution"] > 0 else "lowers"
        st.write(f"- **{r['feature']}** = {r['value']:,.2f} {direction} risk ({r['contribution']:+.2f})")
