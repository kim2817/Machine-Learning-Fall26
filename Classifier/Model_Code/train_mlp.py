
import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, TunedThresholdClassifierCV, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
FOLDER = "Classifier/"
os.makedirs(FOLDER + "Results", exist_ok=True)

# data
df = pd.read_csv(FOLDER + "H2_Cleaned_classification.csv") #load data
#separate target and inputs
X = df.drop(columns=["IsCanceled"]) 
y = df["IsCanceled"]

#group cols by type
categorical = X.select_dtypes(exclude="number").columns.tolist()
numeric = X.select_dtypes(include="number").columns.tolist()

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
#cross validation folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# pipeline
# preprocessing
def make_model(reducer=None, **mlp_params):
    preprocess = ColumnTransformer([ # to unify preprocessing and make each column group go through its own pipeline
        #name,transformer pipeline, column
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),  # fill any missing numbers with the columns median, rescale each column to mean 0 std 1
        numeric), 
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), #fill missing values with the most common category
                          #turn each category into its own 0/1 column
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist",min_frequency=0.005,sparse_output=False))]), 
                          categorical),
     #categories with less than 0.5% merged into single infrequent column
     #any new categories met later goes into infrequent column instead of crashing
     # produce a regular dense array to store every value including zeros
    ])

    steps = [("prep", preprocess)] # to list the steps inside the pipeline
    if reducer is not None:
        steps.append(("reduce", reducer))
    steps.append(("clf", MLPClassifier(max_iter=300, early_stopping=True, n_iter_no_change=20, # wait 20 epochs without improvement so noise doesn't stop training early
                                         random_state=RANDOM_STATE, **mlp_params)))
    return Pipeline(steps)


results = []  # one row per model 


def evaluate(name, features, model):
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": "MLP", "Config": name, "Features": features,
        "CV F1": cv_f1,
        "Test Acc": accuracy_score(y_test, y_pred),
        "Test Prec": precision_score(y_test, y_pred),
        "Test Rec": recall_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred),
        "Test ROC-AUC": roc_auc_score(y_test, y_prob),
        "Train time (s)": train_time,
    })
    print(results[-1])
    return model


# baseline
baseline = evaluate("Baseline", "Original", make_model())

# tuning
param_grid = {
    "clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
    "clf__alpha": [1e-4, 1e-3, 1e-2], #l2 regularization
    "clf__learning_rate_init": [1e-4, 3e-4, 1e-3], # smaller rates are more stable with early stopping
    "clf__batch_size": [64, 256],
    "clf__activation": ["relu", "tanh"],
}
search = RandomizedSearchCV(make_model(), param_grid, n_iter=40, scoring="f1", cv=cv, n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
search.fit(X_train, y_train)  # training set only

best_params = {}
for name, value in search.best_params_.items():
    short_name = name.replace("clf__", "")
    best_params[short_name] = value

print("Best params:", best_params, "CV F1:", round(search.best_score_, 4))

tuned = evaluate("Tuned", "Original", make_model(**best_params))

# decision threshold tuning
# pick cutoff that maximizes F1 using cross validation
tuned_thr = evaluate("Tuned + threshold", "Original",TunedThresholdClassifierCV(make_model(**best_params), scoring="f1", cv=cv, n_jobs=-1))
best_threshold = float(tuned_thr.best_threshold_)
print(f"Best threshold: {best_threshold:.3f}")

# PCA
X_train_prep = make_model().named_steps["prep"].fit_transform(X_train) #build new unfitted model, take the prep step, fit then transform
cum_var = np.cumsum(PCA().fit(X_train_prep).explained_variance_ratio_) #order components from most to least var and gather cumilative var
k = int(np.argmax(cum_var >= 0.95)) + 1  #smallest k with >= 95% var
print(f"PCA: {X_train_prep.shape[1]} features: k = {k} components ")

#plot cumilative variance curve
MLP_IMG_DIR = FOLDER + "Images/MLP/"
os.makedirs(MLP_IMG_DIR, exist_ok=True)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cum_var) + 1), cum_var)
plt.axhline(0.95, color="red", ls="--", label="95% variance")
plt.axvline(k, color="green", ls="--", label=f"k = {k}")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance")
plt.legend()
plt.tight_layout()
plt.savefig(MLP_IMG_DIR + "mlp_pca_explained_variance.png", dpi=150)
plt.close()

# original vs PCA vs LDA
# only feature representation changes
tuned_pca = evaluate("Tuned", f"PCA k={k}", make_model(PCA(n_components=k), **best_params))
tuned_lda = evaluate("Tuned", "LDA",make_model(LinearDiscriminantAnalysis(n_components=1), **best_params))

# results table
comparison = pd.DataFrame(results).round(4)
print(comparison.to_string(index=False))

#figures
print("Tuned:")
print(classification_report(y_test, tuned.predict(X_test), target_names=["Not canceled", "Canceled"]))

y_pred = tuned_thr.predict(X_test)
print(f"Tuned + threshold ({best_threshold:.3f}):")
print(classification_report(y_test, y_pred, target_names=["Not canceled", "Canceled"]))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),display_labels=["Not canceled", "Canceled"]).plot(cmap="Blues")
plt.title(f"MLP tuned confusion matrix (threshold {best_threshold:.2f}, test set)")
plt.tight_layout()
plt.savefig(MLP_IMG_DIR + "mlp_confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, model in [("Baseline", baseline), ("Tuned", tuned), ("Tuned + PCA", tuned_pca), ("Tuned + LDA", tuned_lda)]:
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("MLP ROC curves (test set)")
plt.tight_layout()
plt.savefig(MLP_IMG_DIR + "mlp_roc_curves.png", dpi=150)
plt.close()

# save
RESULTS_DIR = FOLDER + "Results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

#comparison table(baseline/tuned/PCA/LDA)
comparison.to_csv(RESULTS_DIR + "mlp_comparison.csv", index=False)

# every combination tried in the search
pd.DataFrame(search.cv_results_).to_csv(RESULTS_DIR + "mlp_cv_results.csv", index=False)

#search setup + best configuration
with open(RESULTS_DIR + "mlp_best_config.json", "w") as f:
    json.dump({
        "search_method": "RandomizedSearchCV (n_iter=40, 5-fold StratifiedKFold)",
        "optimization_metric": "f1",
        "search_space": {name: [str(v) for v in values] for name, values in param_grid.items()},
        "best_params": best_params,
        "best_cv_f1": float(search.best_score_),
        "pca_k": int(k),
        "best_threshold": best_threshold,
    }, f, indent=2)

# the trained tuned model for the app
joblib.dump(tuned, RESULTS_DIR + "mlp_tuned.joblib")

# input form description for the app (raw columns the model expects)
schema = {} #inputs needed by the model
for col in X.columns:
    if col in categorical:
        schema[col] = {"type": "categorical", "options": sorted(X[col].unique().tolist())}
    else:
        schema[col] = {"type": "numeric", "min": float(X[col].min()), "max": float(X[col].max())}
with open(FOLDER + "Results/feature_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

joblib.dump(baseline, FOLDER + "Model/mlp_baseline.joblib")
joblib.dump(tuned, FOLDER + "Model/mlp_tuned.joblib")
joblib.dump(tuned_pca, FOLDER + "Model/mlp_tuned_pca.joblib")
joblib.dump(tuned_lda, FOLDER + "Model/mlp_tuned_lda.joblib")
joblib.dump(tuned_thr, FOLDER + "Model/mlp_tuned_threshold.joblib")
print("success weeee")
