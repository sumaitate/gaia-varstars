import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

for path in (root / "src", root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gaia_variable_stars.config import (
    CLASSES, CLEAN_FILE, CV_FILE, FEATURE_FILE, FIGURES_ROOT, FOLDS,
    MODEL_FILE, MODELS_ROOT, PRED_FILE, RANK_FILE, SEED, SOURCE_FILE,
)

FE_FEATURE_FILE = FEATURE_FILE.with_name("gaia_features_fe.csv")
FE_VALIDATION_FILE = FEATURE_FILE.with_name("gaia_period_validation_fe.csv")
FE_ORACLE_FILE = FEATURE_FILE.with_name("gaia_features_fe_oracle.csv")
LABEL_FILE = FEATURE_FILE.with_name("gaia_label_confidence.csv")

EXTRA_ROOT = MODELS_ROOT / "gaia_classification"
PLOT_ROOT = FIGURES_ROOT / "gaia_classification"

for path in (MODELS_ROOT, EXTRA_ROOT, PLOT_ROOT):
    path.mkdir(parents=True, exist_ok=True)

PRIMARY = "RF"
REPEATS = 5
BOOTSTRAPS = 1500
PERM_REPEATS = 5
MIN_CELL = 10

GROUPS = {
    "period": [
        "log_period", "power", "half_power", "double_power",
        "half_power_ratio", "double_power_ratio", "fourier_amp", "fourier_ratio",
    ],
    "shape": [
        "magnitude_std", "magnitude_skew", "magnitude_kurt",
        "magnitude_mad", "amplitude", "beyond_1sigma_fraction",
    ],
    "timescale": [
        "von_neumann_eta", "stetson_j", "sf_short",
        "sf_long", "sf_ratio", "max_visit_slope",
    ],
    "color": ["bp_rp_color"],
    "brightness_noise": ["magnitude_mean", "median_error", "amplitude_ratio"],
    "sampling": [
        "n_epochs", "time_span", "median_gap", "short_gap_fraction",
        "n_visits", "epochs_per_visit", "median_visit_gap", "visit_gap_cv",
    ],
}

MAIN_GROUPS = [name for name in GROUPS if name != "sampling"]

def columns(names):
    return [col for name in names for col in GROUPS[name]]

BASE_FEATURES = columns(MAIN_GROUPS)
ALL_FEATURES = columns(GROUPS)

B_STATS = ["magnitude_std", "magnitude_mad", "magnitude_skew", "magnitude_kurt", "amplitude"]

FOURIER_EXTRA = ["R31_naive", "phi21_naive", "phi31_naive"]

C_TIME = B_STATS + GROUPS["period"] + FOURIER_EXTRA

FEATURE_SETS = {
    "full": BASE_FEATURES,
    "oracle": BASE_FEATURES,
    "period_free": columns([name for name in MAIN_GROUPS if name != "period"]),
    "A_metadata": ["bp_rp_color", "magnitude_mean"],
    "B_stats": B_STATS,
    "C_timedomain": C_TIME,
    "D_full": C_TIME + ["bp_rp_color"],
}

for col in ["magnitude_std", "amplitude", "fourier_amp"]:
    FEATURE_SETS[f"drop_{col}"] = [c for c in BASE_FEATURES if c != col]

FAMILY_DROPS = ["drop_color", "drop_brightness_noise", "drop_timescale"]

for family in ["color", "brightness_noise", "timescale"]:
    FEATURE_SETS[f"drop_{family}"] = columns([name for name in MAIN_GROUPS if name != family])

sources = pd.read_csv(SOURCE_FILE)
features = pd.read_csv(FE_FEATURE_FILE)
validation = pd.read_csv(FE_VALIDATION_FILE)
oracle = pd.read_csv(FE_ORACLE_FILE)
labels = pd.read_csv(LABEL_FILE)

feature_cols = columns(["period", "shape", "timescale", "brightness_noise"]) + FOURIER_EXTRA

stars = (
    sources[["source_id", "class_name", "bp_rp_color", "g_mean_magnitude", "gaia_period"]]
    .merge(
        validation[["source_id", "period_eligible", "period_status", "sky_cell", *GROUPS["sampling"]]],
        on="source_id", how="left", validate="one_to_one",
    )
    .merge(features[["source_id", *feature_cols]], on="source_id", how="left", validate="one_to_one")
    .merge(
        labels[["source_id", "matches_general_class", "best_class_score"]],
        on="source_id", how="left", validate="one_to_one",
    )
    .sort_values("source_id")
    .reset_index(drop=True)
)

if len(stars) != 2000 or stars["sky_cell"].isna().any():
    raise ValueError("Expected 2,000 stars with period-validation and sky-cell data.")

excluded = len(stars) - len(features)
stars = stars.loc[stars["source_id"].isin(features["source_id"])].reset_index(drop=True)
assert len(stars) == len(features)

stars["matches_general_class"] = stars["matches_general_class"].astype(bool)

numeric = list(dict.fromkeys([*ALL_FEATURES, *FOURIER_EXTRA, "g_mean_magnitude", "gaia_period"]))
stars[numeric] = stars[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
stars["eligible"] = stars["period_eligible"].fillna(False).astype(bool)

oracle_stars = stars.copy()
reference = oracle.set_index("source_id").reindex(stars.loc[stars["eligible"], "source_id"])

if reference["log_period"].isna().any():
    raise ValueError("Oracle periods are missing for eligible stars.")

oracle_stars.loc[stars["eligible"], GROUPS["period"]] = reference[GROUPS["period"]].to_numpy()

print("Total stars:", len(stars))
print("Excluded before modeling (not searchable):", excluded)

def make_model(name, variant):
    if name == "Dummy":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", DummyClassifier(strategy="stratified", random_state=SEED)),
        ])

    if name == "LR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=SEED)),
        ])

    if name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            )),
            ("model", RandomForestClassifier(
                n_estimators=250, min_samples_leaf=2,
                max_features="sqrt", random_state=SEED, n_jobs=-1
            )),
        ])

    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, min_samples_leaf=15,
        l2_regularization=1.0, random_state=SEED,
    )
    return Pipeline([
        ("flags", FeatureUnion([
            ("values", FunctionTransformer()),
            ("missing", MissingIndicator(features="all", sparse=False)),
        ])),
        ("model", model),
    ])


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["split", "evaluate"])
args = parser.parse_args()

SPLIT_ROOT = root / "splits"
RESULT_ROOT = MODELS_ROOT / "locked_test"

SPLIT_ROOT.mkdir(exist_ok=True)
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

names = ["train", "validation", "test"]
paths = {
    name: SPLIT_ROOT / f"{name}_source_ids.csv"
    for name in names
}

# Create permanent 60/20/20 grouped partitions.
if args.action == "split":

    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Locked splits already exist.")

    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )

    partitions = [
        test for _, test in cv.split(
            stars, stars["class_name"], stars["sky_cell"]
        )
    ]

    indices = {
        "train": np.concatenate(partitions[2:]),
        "validation": partitions[1],
        "test": partitions[0],
    }

    for name, idx in indices.items():
        assert set(stars.iloc[idx]["class_name"]) == set(CLASSES)

    for name, idx in indices.items():
        stars.iloc[idx][["source_id"]].sort_values(
            "source_id"
        ).to_csv(paths[name], index=False)

        print(name, len(idx))

    print("SPLITS LOCKED")
    sys.exit(0)

# Load the permanent partitions.
data = stars.set_index("source_id")

parts = {
    name: data.loc[pd.read_csv(paths[name])["source_id"]]
    for name in names
}

train = parts["train"]
validation = parts["validation"]
test = parts["test"]

# Verify source and spatial independence.
combined = pd.concat(list(parts.values()))

assert combined.index.is_unique
assert set(combined.index) == set(data.index)

for i, a in enumerate(names):
    assert set(parts[a]["class_name"]) == set(CLASSES)

    for b in names[i + 1:]:
        assert set(parts[a]["sky_cell"]).isdisjoint(
            parts[b]["sky_cell"]
        )

def metrics(truth, predicted):
    return {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(
            truth, predicted, labels=CLASSES,
            average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            truth, predicted
        ),
    }

# Evaluate baselines and ablations on validation data only.
rows = []

for name in ["Dummy", "LR", "RF", "HGB"]:
    variants = ["full"]

    if name in ["RF", "HGB"]:
        variants += [
            "A_metadata", "B_stats",
            "C_timedomain", "D_full"
        ]

    for variant in variants:
        cols = FEATURE_SETS[variant]
        model = make_model(name, variant)

        model.fit(train[cols], train["class_name"])
        predicted = model.predict(validation[cols])

        rows.append({
            "model": name,
            "feature_set": variant,
            **metrics(validation["class_name"], predicted)
        })

pd.DataFrame(rows).to_csv(
    RESULT_ROOT / "validation_metrics.csv", index=False
)

# Refit the predetermined primary Random Forest.
development = pd.concat([train, validation])

model = make_model(PRIMARY, "full")
model.fit(development[BASE_FEATURES], development["class_name"])

# Final locked test evaluation.
truth = test["class_name"].to_numpy()
predicted = model.predict(test[BASE_FEATURES])

point = metrics(truth, predicted)

# HEALPix-grouped bootstrap: 1,500 replicates.
sky = test["sky_cell"].to_numpy()
groups = [
    np.flatnonzero(sky == cell)
    for cell in np.unique(sky)
]

rng = np.random.default_rng(SEED)
draws = []

while len(draws) < 1500:
    sampled = rng.integers(len(groups), size=len(groups))
    idx = np.concatenate([groups[i] for i in sampled])

    if set(truth[idx]) != set(CLASSES):
        continue

    draws.append(metrics(truth[idx], predicted[idx]))

boot = pd.DataFrame(draws)

results = pd.DataFrame({
    "metric": list(point),
    "score": list(point.values()),
    "ci_lower": [boot[k].quantile(0.025) for k in point],
    "ci_upper": [boot[k].quantile(0.975) for k in point],
})

results.to_csv(RESULT_ROOT / "test_metrics.csv", index=False)

pd.DataFrame({
    "source_id": test.index,
    "true_class": truth,
    "predicted_class": predicted,
}).to_csv(RESULT_ROOT / "test_predictions.csv", index=False)

pd.DataFrame(
    confusion_matrix(truth, predicted, labels=CLASSES),
    index=CLASSES,
    columns=CLASSES,
).to_csv(RESULT_ROOT / "confusion_matrix.csv")

p, r, f, s = precision_recall_fscore_support(
    truth, predicted, labels=CLASSES, zero_division=0
)

pd.DataFrame({
    "class_name": CLASSES,
    "precision": p,
    "recall": r,
    "f1": f,
    "support": s,
}).to_csv(RESULT_ROOT / "per_class.csv", index=False)

joblib.dump(model, RESULT_ROOT / "model.joblib")

print(results.to_string(index=False))
print("LOCKED TEST EVALUATION COMPLETE")
