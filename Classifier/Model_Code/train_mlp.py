
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
from sklearn.metrics import *
from sklearn.model_selection import *
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
    steps.append(("clf", MLPClassifier(max_iter=300, early_stopping=True,random_state=RANDOM_STATE, **mlp_params)))
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
