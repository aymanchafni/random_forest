"""Heart-disease classification: random forests vs. baselines, evaluated properly.

    python train.py      # writes docs/*.png and docs/results.md (~1 min)
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

SEED = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TARGET = "HeartDisease"


def load() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ROOT, "heart.csv"))
    # 172 patients (19 %) have Cholesterol = 0 and one has RestingBP = 0: physiologically
    # impossible, these are missing values. Treat them as such (imputed + flagged) instead of
    # letting the model learn from fake zeros.
    df[["Cholesterol", "RestingBP"]] = df[["Cholesterol", "RestingBP"]].replace(0, np.nan)
    return df


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def models() -> dict[str, Pipeline]:
    rf = lambda **kw: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **kw)  # noqa: E731
    return {
        "Majority class (baseline)": Pipeline([("prep", preprocessor(False)),
                                               ("clf", DummyClassifier(strategy="most_frequent"))]),
        "Logistic regression": Pipeline([("prep", preprocessor(True)), ("clf", LogisticRegression(max_iter=1000))]),
        "Random forest, depth 2 (original)": Pipeline([("prep", preprocessor(False)), ("clf", rf(max_depth=2))]),
        "Random forest, tuned": GridSearchCV(
            Pipeline([("prep", preprocessor(False)), ("clf", rf())]),
            param_grid={"clf__max_depth": [None, 4, 8], "clf__min_samples_leaf": [1, 3, 5],
                        "clf__max_features": ["sqrt", 0.5]},
            scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1,
        ),
    }


def main():
    os.makedirs(DOCS, exist_ok=True)
    df = load()
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "recall": "recall", "roc_auc": "roc_auc"}
    rows, fitted = [], {}
    for name, model in models().items():
        # Cross-validation on the training set (for the tuned forest this is nested CV:
        # the grid search runs inside every outer fold, so the estimate is not optimistic).
        cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        model.fit(X_train, y_train)
        fitted[name] = model
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "cv_auc": f"{cvres['test_roc_auc'].mean():.3f} ± {cvres['test_roc_auc'].std():.3f}",
            "cv_accuracy": f"{cvres['test_accuracy'].mean():.3f} ± {cvres['test_accuracy'].std():.3f}",
            "test_accuracy": accuracy_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_auc": roc_auc_score(y_test, proba),
        })
        print(f"{name:<36} CV AUC {rows[-1]['cv_auc']}  test AUC {rows[-1]['test_auc']:.3f}")

    best_name = "Random forest, tuned"
    best = fitted[best_name]

    # ROC curves on the held-out test set
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        if "baseline" not in name:
            RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.set_title("ROC curves (held-out test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "roc.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix + permutation importance for the tuned forest
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})
    ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, display_labels=["Normal", "Heart disease"],
                                          cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title("Tuned random forest: confusion matrix")
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
    order = np.argsort(imp.importances_mean)
    axes[1].barh(X_test.columns[order], imp.importances_mean[order], xerr=imp.importances_std[order], color="#2563eb")
    axes[1].set_title("Permutation importance (drop in test ROC AUC)")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "evaluation.png"), dpi=150)
    plt.close(fig)

    # One tree of the shallow forest (no Graphviz needed)
    shallow = fitted["Random forest, depth 2 (original)"]
    names = shallow.named_steps["prep"].get_feature_names_out()
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_tree(shallow.named_steps["clf"].estimators_[0], feature_names=names, class_names=["Normal", "Disease"],
              filled=True, rounded=True, impurity=False, proportion=True, ax=ax, fontsize=9)
    fig.savefig(os.path.join(DOCS, "tree.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    params = {k.replace("clf__", ""): v for k, v in best.best_params_.items()}
    table = pd.DataFrame(rows)
    lines = ["# Results", "", f"Train {len(X_train)} / test {len(X_test)} patients (stratified, seed {SEED}).", "",
             "| Model | CV ROC AUC | CV accuracy | Test accuracy | Test recall | Test precision | Test ROC AUC |",
             "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.model} | {r.cv_auc} | {r.cv_accuracy} | {r.test_accuracy:.3f} | {r.test_recall:.3f} "
                     f"| {r.test_precision:.3f} | {r.test_auc:.3f} |")
    lines += ["", f"Tuned forest hyper-parameters: `{params}`", ""]
    top = X_test.columns[order[::-1][:5]].tolist()
    lines.append(f"Most important features (permutation, test set): {', '.join(top)}")
    with open(os.path.join(DOCS, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
