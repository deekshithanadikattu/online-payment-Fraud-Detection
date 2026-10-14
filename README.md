# Online Payment Fraud Detection

XGBoost model that flags likely fraudulent mobile-money transactions (PaySim dataset), with a
Streamlit app that scores a transaction and explains which factors drove the decision.

## What makes this version different
- **Evaluated on the future, not a random shuffle.** Train on the first 60% of the timeline,
  tune on the next 20%, report on the final 20%.
- **Business-cost threshold.** A missed fraud costs its full amount, and a false alarm costs a
  manual review. The decision threshold minimises that total cost on the validation period.
- **Honest metrics.** Fraud is a tiny fraction of transactions, so accuracy is meaningless
  (predicting "legitimate" every time scores above 99%). Reported instead: PR-AUC, precision,
  recall, share of fraud dollars caught, and share of transactions flagged.
- **No account IDs as features.** They can't generalise to new customers.
- **Explainable.** The app shows each prediction's top contributing features (SHAP values).

## Run it
```bash
pip install pandas numpy xgboost scikit-learn streamlit
python train.py --data data.csv      # writes fraud_model.json + model_meta.json
streamlit run app.py
```
`--review-cost` sets the assumed cost of reviewing one flagged transaction (default 25).

## Results
Fill in from `model_meta.json` → `metrics.test` after training on the full dataset.
