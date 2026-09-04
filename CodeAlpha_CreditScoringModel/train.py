from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

SEED = 42
FEATURES = [
    "annual_income", "monthly_debt", "credit_utilization",
    "missed_payments_12m", "credit_history_years", "employment_years",
    "existing_loans", "savings_balance", "age"
]


def make_credit_dataset(n=6000, seed=SEED):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    annual_income = np.clip(rng.lognormal(np.log(55000), 0.55, n), 15000, 250000)
    monthly_debt = np.clip(rng.gamma(2.0, 500, n), 0, 6500)
    credit_utilization = np.clip(rng.beta(2.1, 3.2, n), 0, 1)
    missed_payments_12m = np.clip(rng.poisson(0.8, n), 0, 8)
    credit_history_years = np.clip((age - 18) * rng.uniform(0.25, 0.95, n), 0.5, 45)
    employment_years = np.clip(rng.gamma(2.5, 2.5, n), 0, age - 18)
    existing_loans = np.clip(rng.poisson(1.5, n), 0, 7)
    savings_balance = np.clip(rng.lognormal(np.log(9000), 1.0, n), 0, 200000)

    monthly_income = annual_income / 12
    debt_to_income = monthly_debt / np.maximum(monthly_income, 1)
    savings_ratio = savings_balance / np.maximum(annual_income, 1)

    risk = (
        2.8 * credit_utilization
        + 2.2 * debt_to_income
        + 0.55 * missed_payments_12m
        + 0.16 * existing_loans
        - 0.045 * credit_history_years
        - 0.035 * employment_years
        - 1.3 * savings_ratio
        - 0.000004 * annual_income
        + rng.normal(0, 0.65, n)
    )
    threshold = np.quantile(risk, 0.48)
    creditworthy = (risk < threshold).astype(int)

    return pd.DataFrame({
        "annual_income": annual_income,
        "monthly_debt": monthly_debt,
        "credit_utilization": credit_utilization,
        "missed_payments_12m": missed_payments_12m,
        "credit_history_years": credit_history_years,
        "employment_years": employment_years,
        "existing_loans": existing_loans,
        "savings_balance": savings_balance,
        "age": age,
        "creditworthy": creditworthy,
    })


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
    }, pred, prob


def main():
    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    df = make_credit_dataset()
    X, y = df[FEATURES], df["creditworthy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    numeric = ColumnTransformer([("num", StandardScaler(), FEATURES)])
    models = {
        "Logistic Regression": Pipeline([
            ("prep", numeric),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=15, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=SEED
        ),
    }

    results, fitted = [], {}
    best = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, pred, prob = evaluate(name, model, X_test, y_test)
        results.append(metrics)
        fitted[name] = (model, pred, prob)
        if best is None or metrics["roc_auc"] > best["metrics"]["roc_auc"]:
            best = {"name": name, "metrics": metrics, "model": model, "pred": pred, "prob": prob}

    joblib.dump(best["model"], out / "best_credit_model.joblib")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    schema = {}
    for col in FEATURES:
        s = df[col]
        schema[col] = {
            "min": float(s.quantile(0.01)), "max": float(s.quantile(0.99)),
            "default": float(s.median())
        }
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    best_prob = best["prob"]
    fpr, tpr, _ = roc_curve(y_test, best_prob)
    fig = plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{best['name']} (AUC={best['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Credit Scoring ROC Curve")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=180)
    plt.close(fig)

    cm = confusion_matrix(y_test, best["pred"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Not creditworthy", "Creditworthy"]).plot(ax=ax)
    plt.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(pd.DataFrame(results).sort_values("roc_auc", ascending=False).to_string(index=False))
    print(f"\nBest model: {best['name']}")


if __name__ == "__main__":
    main()
