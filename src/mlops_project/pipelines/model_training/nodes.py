import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import re

import mlflow
import mlflow.sklearn


def manual_feature_selection(df: pd.DataFrame):
    """ Manually removes problematic features from feature set (target, ids, leakage, object)"""
    
    target = "mortality_severity"

    leakage_cols = [
        "mortality_rate",
        "mortality_severity",
        "Deaths",
        "Confirmed",
        "Recovered",
        "Active",
        "deaths_per_100k",
        "active_cases_per_100k",
        "confirmed_per_100k",
        "recovery_rate",
    ]

    identifier_cols = [
        "country",
    ]

    X = df.drop(columns=leakage_cols + identifier_cols, errors="ignore")
    y = df[target]

    X = X.select_dtypes(include=["number", "category", "object"])

    # Keep Region, but one-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def random_forest_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 30,
):
    """ Uses random forest feature importance to select relevant features for prediction"""
    rf_selector = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    rf_selector.fit(X, y)

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": rf_selector.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    selected_features = feature_importance.head(top_n)["feature"].tolist()

    return selected_features, feature_importance


def train_models(df: pd.DataFrame):
    X, y = manual_feature_selection(df)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    selected_features, feature_importance = random_forest_feature_selection(
        X, y_encoded, top_n=30
    )

    X = X[selected_features]
    X.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', col)
        for col in X.columns
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            eval_metric="mlogloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            min_data_in_leaf=10,
            random_state=42
        ),
    }

    mlflow.set_experiment("covid_mortality_prediction")

    trained_models = {}
    metrics_rows = []

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1)


            trained_models[model_name] = model

            metrics_rows.append(
                {
                    "model": model_name,
                    "accuracy": acc,
                    "f1_macro": f1,
                }
            )

    metrics = pd.DataFrame(metrics_rows)

    best_model_name = metrics.sort_values(by="f1_macro", ascending=False).iloc[0]["model"]
    best_model = trained_models[best_model_name]

    return trained_models, metrics, feature_importance, best_model