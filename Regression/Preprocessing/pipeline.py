import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

TARGET = "ADR"

cat_cols = ["ReservedRoomType", "Meal", "Country", "MarketSegment", "DistributionChannel",
            "CustomerType", "DepositType", "ArrivalDateMonth"]
skewed_cols = ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights"]
num_cols = ["Adults", "Children", "Babies", "ArrivalDateYear",
            "PreviousCancellations", "PreviousBookingsNotCanceled"]
binary_cols = ["IsRepeatedGuest", "HasCompany"]


def check_groups(X):
    # every predictor must be in exactly one group
    groups = cat_cols + skewed_cols + num_cols + binary_cols
    assert sorted(groups) == sorted(X.columns), set(X.columns) ^ set(groups)


def build_preprocess():
    # unfitted, it gets fitted on the training data only
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005), cat_cols),
        ("skewed", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("sc", StandardScaler())]), skewed_cols),
        ("num", StandardScaler(), num_cols),
        ("bin", "passthrough", binary_cols),
    ])
