from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, t, ttest_ind


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
TARGET = "Churn"
IDENTIFIER = "customerID"
AUDIT_COLUMNS = ["gender", "SeniorCitizen"]
EXCLUDED_COLUMNS = [IDENTIFIER, TARGET, "gender", "SeniorCitizen", "Partner", "Dependents"]
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


@dataclass
class ModelBundle:
    data: pd.DataFrame
    model: CalibratedClassifierCV
    naive_bayes: Pipeline
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    scores: pd.DataFrame
    metrics: pd.DataFrame
    test_probabilities: dict[str, np.ndarray]


def load_data() -> pd.DataFrame:
    """Read the bundled IBM sample and repair its numeric total-charge field."""
    frame = pd.read_csv(DATA_PATH)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame[TARGET] = frame[TARGET].astype(str).str.strip()
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in EXCLUDED_COLUMNS]


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [column for column in columns if column in NUMERIC_COLUMNS]
    categorical = [column for column in columns if column not in numeric]
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        verbose_feature_names_out=True,
    )


def make_logistic_pipeline(columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_preprocessor(columns)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def _metric_row(name: str, y_true: pd.Series, probabilities: np.ndarray, capacity: float = 0.20) -> dict[str, Any]:
    order = np.argsort(-probabilities)
    top_count = max(1, int(np.ceil(len(order) * capacity)))
    selected = order[:top_count]
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "Model": name,
        "ROC AUC": roc_auc_score(y_true, probabilities),
        "Average precision": average_precision_score(y_true, probabilities),
        "Brier score ↓": brier_score_loss(y_true, probabilities),
        "Recall at 20% capacity": float(np.asarray(y_true)[selected].sum()) / max(float(np.asarray(y_true).sum()), 1),
        "Precision @ 0.50": precision_score(y_true, predicted, zero_division=0),
    }


def train_models(frame: pd.DataFrame) -> ModelBundle:
    columns = feature_columns(frame)
    x = frame[columns].copy()
    y = frame[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    base = make_logistic_pipeline(columns)
    calibrated = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5, n_jobs=1)
    calibrated.fit(x_train, y_train)

    naive_bayes = Pipeline(
        [("prep", make_preprocessor(columns)), ("model", GaussianNB())]
    )
    naive_bayes.fit(x_train, y_train)

    test_probabilities = {
        "Calibrated logistic regression": calibrated.predict_proba(x_test)[:, 1],
        "Gaussian Naive Bayes": naive_bayes.predict_proba(x_test)[:, 1],
    }
    metrics = pd.DataFrame(
        [_metric_row(label, y_test, values) for label, values in test_probabilities.items()]
    )

    # Use only held-out rows for the bundled portfolio view. Scoring training
    # examples would make the demo queue look more certain than its evaluation.
    test_rows = frame.loc[x_test.index]
    scores = test_rows[[IDENTIFIER, "MonthlyCharges", "tenure", "Contract", "InternetService", TARGET]].copy()
    heldout_probabilities = pd.Series(
        test_probabilities["Calibrated logistic regression"], index=x_test.index
    ).reindex(test_rows.index)
    scores["risk_probability"] = heldout_probabilities.to_numpy()
    scores["risk_band"] = pd.cut(
        scores["risk_probability"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Watch", "High", "Very high"],
        ordered=True,
    ).astype(str)
    scores["risk_weighted_charges"] = scores["risk_probability"] * scores["MonthlyCharges"]
    scores["actual_churn"] = scores[TARGET].eq("Yes")
    scores.drop(columns=[TARGET], inplace=True)

    return ModelBundle(
        data=frame,
        model=calibrated,
        naive_bayes=naive_bayes,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=columns,
        scores=scores,
        metrics=metrics,
        test_probabilities=test_probabilities,
    )


def model_signals(bundle: ModelBundle) -> pd.DataFrame:
    """Return average absolute logistic coefficients grouped by source field."""
    weights = []
    for calibrated_model in bundle.model.calibrated_classifiers_:
        pipeline = calibrated_model.estimator
        prep = pipeline.named_steps["prep"]
        coefficients = pipeline.named_steps["model"].coef_[0]
        names = prep.get_feature_names_out()
        part = pd.DataFrame({"encoded": names, "weight": np.abs(coefficients)})
        part["feature"] = part["encoded"].map(lambda value: _source_feature(value, bundle.feature_columns))
        weights.append(part.groupby("feature", as_index=False)["weight"].mean())
    combined = pd.concat(weights, ignore_index=True)
    return (
        combined.groupby("feature", as_index=False)["weight"]
        .mean()
        .sort_values("weight", ascending=True)
        .tail(10)
    )


def explain_customer(bundle: ModelBundle, row: pd.DataFrame) -> pd.DataFrame:
    """Explain a score as average feature contribution to log odds vs the training mix."""
    contributions: list[pd.DataFrame] = []
    for calibrated_model in bundle.model.calibrated_classifiers_:
        pipeline = calibrated_model.estimator
        prep = pipeline.named_steps["prep"]
        encoded_train = prep.transform(bundle.x_train)
        encoded_row = prep.transform(row[bundle.feature_columns])
        coefficients = pipeline.named_steps["model"].coef_[0]
        names = prep.get_feature_names_out()
        centered = (encoded_row - np.asarray(encoded_train).mean(axis=0)).ravel() * coefficients
        part = pd.DataFrame({"encoded": names, "contribution": centered})
        part["feature"] = part["encoded"].map(lambda value: _source_feature(value, bundle.feature_columns))
        contributions.append(part.groupby("feature", as_index=False)["contribution"].sum())
    combined = pd.concat(contributions, ignore_index=True)
    return (
        combined.groupby("feature", as_index=False)["contribution"]
        .mean()
        .sort_values("contribution", key=lambda values: values.abs(), ascending=False)
        .head(8)
    )


def _source_feature(encoded_name: str, columns: list[str]) -> str:
    name = encoded_name.split("__", maxsplit=1)[-1]
    matches = [column for column in columns if name == column or name.startswith(f"{column}_")]
    return max(matches, key=len) if matches else name


def audit_slices(bundle: ModelBundle) -> pd.DataFrame:
    probabilities = bundle.test_probabilities["Calibrated logistic regression"]
    threshold = 0.50
    predicted = probabilities >= threshold
    test = bundle.data.loc[bundle.x_test.index]
    rows = []
    for column in AUDIT_COLUMNS:
        if column == "SeniorCitizen":
            labels = test[column].map({0: "Not senior", 1: "Senior"})
        else:
            labels = test[column].astype(str)
        for label in sorted(labels.dropna().unique()):
            mask = labels.to_numpy() == label
            if mask.sum() == 0:
                continue
            actual = bundle.y_test.to_numpy()[mask]
            pred = predicted[mask]
            tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "Review slice": f"{column}: {label}",
                    "Customers": int(mask.sum()),
                    "Observed churn": float(actual.mean()),
                    "Recall @ 0.50": float(tp / max(tp + fn, 1)),
                    "False positive rate": float(fp / max(fp + tn, 1)),
                    "Brier score": brier_score_loss(actual, probabilities[mask]),
                }
            )
    return pd.DataFrame(rows)


def calibration_bins(bundle: ModelBundle, bins: int = 10) -> pd.DataFrame:
    probabilities = bundle.test_probabilities["Calibrated logistic regression"]
    result = pd.DataFrame({"probability": probabilities, "actual": bundle.y_test.to_numpy()})
    result["band"] = pd.cut(result["probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    grouped = result.groupby("band", observed=True).agg(
        predicted=("probability", "mean"), observed=("actual", "mean"), customers=("actual", "size")
    )
    return grouped.reset_index(drop=True)


def statistical_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Screen univariate sample associations, then control false discovery rate."""
    rows: list[dict[str, Any]] = []
    churned = frame[TARGET].eq("Yes")
    for column in frame.columns:
        if column in {TARGET, IDENTIFIER}:
            continue
        series = frame[column]
        if column in NUMERIC_COLUMNS:
            positive = pd.to_numeric(series[churned], errors="coerce").dropna().to_numpy()
            negative = pd.to_numeric(series[~churned], errors="coerce").dropna().to_numpy()
            if len(positive) < 2 or len(negative) < 2:
                continue
            test = ttest_ind(positive, negative, equal_var=False, nan_policy="omit")
            difference = float(positive.mean() - negative.mean())
            variance_positive = float(positive.var(ddof=1))
            variance_negative = float(negative.var(ddof=1))
            se_squared = variance_positive / len(positive) + variance_negative / len(negative)
            if se_squared <= 0:
                continue
            degrees = se_squared**2 / (
                (variance_positive / len(positive)) ** 2 / max(len(positive) - 1, 1)
                + (variance_negative / len(negative)) ** 2 / max(len(negative) - 1, 1)
            )
            margin = float(t.ppf(.975, degrees) * np.sqrt(se_squared))
            rows.append(
                {
                    "Feature": column,
                    "Test": "Welch t-test",
                    "Effect": difference,
                    "Effect detail": "Mean difference (churned − retained)",
                    "95% CI": f"[{difference - margin:.2f}, {difference + margin:.2f}]",
                    "Effect size": abs(difference) / max(float(np.sqrt((variance_positive + variance_negative) / 2)), 1e-12),
                    "p-value": float(test.pvalue),
                }
            )
        else:
            table = pd.crosstab(series.fillna("Missing"), churned)
            if min(table.shape) < 2:
                continue
            chi2, p_value, _, _ = chi2_contingency(table)
            n = int(table.to_numpy().sum())
            dof_min = min(table.shape[0] - 1, table.shape[1] - 1)
            cramers_v = float(np.sqrt(chi2 / max(n * dof_min, 1)))
            rows.append(
                {
                    "Feature": column,
                    "Test": "Chi-square",
                    "Effect": cramers_v,
                    "Effect detail": "Cramér's V",
                    "95% CI": "—",
                    "Effect size": cramers_v,
                    "p-value": float(p_value),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    p_values = result["p-value"].to_numpy()
    order = np.argsort(p_values)
    adjusted = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    result["BH q-value"] = 1.0
    result.loc[result.index[order], "BH q-value"] = adjusted
    result["Evidence"] = np.where(result["BH q-value"] < .05, "q < 0.05", "Not below 0.05")
    return result.sort_values("BH q-value", ascending=True).reset_index(drop=True)


def experiment_assignment(
    eligible: pd.DataFrame, holdout_share: float, seed: int
) -> pd.DataFrame:
    """Randomize within risk bands while preserving the existing priority rank."""
    assigned = eligible.copy()
    assigned["experiment_arm"] = "Treatment"
    if assigned.empty:
        return assigned
    assigned["_block"] = pd.qcut(
        assigned["risk_probability"].rank(method="first"),
        q=min(5, max(1, len(assigned) // 2)),
        labels=False,
        duplicates="drop",
    )
    rng = np.random.default_rng(seed)
    groups = list(assigned.groupby("_block", observed=True))
    counts = [len(group) for _, group in groups]
    if len(assigned) > 1:
        target_holdout = min(max(int(round(len(assigned) * holdout_share)), 1), len(assigned) - 1)
    else:
        target_holdout = 0
    quotas = [int(np.floor(count * holdout_share)) for count in counts]
    remainder_order = sorted(
        range(len(groups)),
        key=lambda index: counts[index] * holdout_share - quotas[index],
        reverse=True,
    )
    remaining = target_holdout - sum(quotas)
    for index in remainder_order:
        if remaining <= 0:
            break
        if quotas[index] < counts[index] - 1:
            quotas[index] += 1
            remaining -= 1
    for group_index, (_, group) in enumerate(groups):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        assigned.loc[indices[: quotas[group_index]], "experiment_arm"] = "Holdout"
    return assigned.drop(columns=["_block"])

