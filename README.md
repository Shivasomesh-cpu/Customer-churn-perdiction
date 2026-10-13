# StaySignal

**A churn-risk workbench that helps teams decide whom to review, what the model saw, and how to test a retention action.**

StaySignal develops the probability and statistical-learning project in the accompanying class deck into a runnable local application. It trains on IBM's public Telco Customer Churn sample and includes a scored portfolio, a capacity-aware triage queue, individual explanations, model diagnostics, a fairness review, and a randomized holdout planner.

## Start here

### Windows

1. Install Python 3.10 or newer.
2. Double-click `run.bat`, or run these commands in PowerShell:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser at the local address printed by Streamlit. The first launch trains and evaluates the models; later page changes reuse the result in the running session.

## What it does

- **Portfolio view:** churn rate, risk distribution, customer mix, and the model's leading signals.
- **CSV scoring:** upload a current active-account export using the supplied template. Required columns are `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`. `customerID` is optional, but IDs must be unique when supplied; demographic fields are ignored. Blank input values are imputed, monthly charges must be valid non-negative numbers, and unseen categories are flagged and treated as unknown.
- **Triage queue:** prioritize a limited number of active-account reviews by calibrated churn probability multiplied by monthly charges. Filter by contract and risk band, then export a CSV.
- **Customer view:** score one account, see its probability, and inspect the strongest positive and negative feature contributions from the logistic model.
- **Experiment planner:** randomize the selected queue into treatment and holdout groups within risk bands, export assignments, and estimate a rough minimum detectable difference for the planned sample size.
- **Statistical evidence:** screen numeric fields with Welch's t-test and categorical fields with chi-square, report effect sizes and numeric confidence intervals, and adjust p-values with Benjamini–Hochberg.
- **Model review:** compare logistic regression and Gaussian Naive Bayes on a stratified holdout; inspect ROC AUC, average precision, Brier score, recall at 20% capacity, a calibration plot, and demographic audit slices.

## Modeling choices and limits

The target is the snapshot's existing `Churn` label. This is a retrospective classification exercise, not a validated forecast over a specified future period. The app uses a stratified 80/20 train/test split and sigmoid probability calibration inside the training partition. The test partition is kept out of model fitting and calibration; bundled account scores and evaluation metrics use only this held-out partition.

`customerID`, gender, senior-citizen status, partner status, and dependent status are excluded from prediction. Gender and senior-citizen status remain available only for aggregate review; this does not prove the model is fair. Other service and billing attributes can still act as proxies.

For an uploaded current-account CSV, all 15 model fields listed above are required. The app ignores a `Churn` column if one is included and assumes uploaded rows are active accounts. The bundled template contains example rows to show the field names and format.

The queue's **risk-weighted charge score** is `probability of the sample Churn label × monthly charges`. Because the label has no defined future time window, this is only a prioritization proxy—not expected monthly loss, savings, lifetime value, or margin. The sample has no retention treatment outcomes, so StaySignal does **not** claim an offer will save any particular customer. The experiment planner is designed to help collect randomized evidence before making that claim. Its minimum-detectable-effect calculation is an approximate planning aid, not a power guarantee.

This project is an educational demonstration built on a fictional telco sample. Review data quality, consent, local policy, fairness, and model performance on current data before any operational use.

## Project map

```text
app.py                         Streamlit application
src/staysignal/engine.py       Data loading, model training, evaluation, scoring
data/telco_customer_churn.csv  IBM sample dataset
data/IBM_REPOSITORY_LICENSE.txt
requirements.txt
run.bat
```

## Data and research notes

- Dataset: [IBM Telco Customer Churn sample and modeling repository](https://github.com/IBM/telco-customer-churn-on-icp4d). The repository describes data analysis and model building for a fictional telco. The CSV is included so the demo works without a separate download.
- Probability calibration: [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration). It recommends evaluating probability quality and fitting calibration independently of the final test data; this project uses a training-only cross-validated sigmoid calibrator and reports Brier score plus a reliability plot.
- Risk versus treatment effect: [Why you should stop predicting customer churn and start using uplift models](https://www.sciencedirect.com/science/article/pii/S0020025519312022) and [Uplift modelling and its implications for B2B customer churn prediction](https://www.sciencedirect.com/science/article/pii/S0019850121001930). A churn score estimates propensity; it cannot establish which intervention changes behavior. StaySignal therefore offers an experiment planner rather than an unsupported offer recommender.

## License

Application code in this project is released under the MIT License. The IBM repository's Apache 2.0 license is retained at `data/IBM_REPOSITORY_LICENSE.txt`; see the source repository for the dataset's provenance and terms.
