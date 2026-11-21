# for data manipulation
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("engine-maintenance-training-experiment")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# All six engine sensor features are numeric, so scale them
numeric_features = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
                    "Coolant pressure", "lub oil temp", "Coolant temp"]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Define the two models and their parameter grids.
# Small grids so the pipeline runs fast on GitHub Actions.
# Widen these if you want a more thorough hyperparameter search.
models = {
    "random_forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "randomforestclassifier__n_estimators": [100, 200],
            "randomforestclassifier__max_depth": [5, 10],
            "randomforestclassifier__min_samples_leaf": [1, 5],
        },
    ),
    "xgboost": (
        xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        {
            "xgbclassifier__n_estimators": [50, 100],
            "xgbclassifier__max_depth": [2, 3],
            "xgbclassifier__learning_rate": [0.05, 0.1],
        },
    ),
}

classification_threshold = 0.5

best_overall_score, best_overall_name, best_overall_model = -1, None, None

# Start MLflow run
with mlflow.start_run(run_name="model_comparison"):
    for name, (model, param_grid) in models.items():
        print(f"\n===== {name} =====")
        model_pipeline = make_pipeline(preprocessor, model)

        with mlflow.start_run(run_name=name, nested=True):
            # Hyperparameter tuning (F1 balances recall and precision for the Faulty class)
            grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
            grid_search.fit(Xtrain, ytrain)

            # Log all parameter combinations and their mean test scores
            results = grid_search.cv_results_
            for i in range(len(results["params"])):
                param_set = results["params"][i]
                mean_score = results["mean_test_score"][i]
                std_score = results["std_test_score"][i]

                # Log each combination as a separate MLflow run
                with mlflow.start_run(nested=True):
                    mlflow.log_params(param_set)
                    mlflow.log_metric("mean_test_score", mean_score)
                    mlflow.log_metric("std_test_score", std_score)

            # Log best parameters separately for this model
            mlflow.log_param("model_name", name)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_f1", grid_search.best_score_)

            # Store and evaluate the best model
            best_model = grid_search.best_estimator_
            print("Best params:", grid_search.best_params_)

            y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
            y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

            y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
            y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

            train_report = classification_report(ytrain, y_pred_train, output_dict=True)
            test_report = classification_report(ytest, y_pred_test, output_dict=True)
            print(classification_report(ytest, y_pred_test))

            # Log the metrics for the best model
            mlflow.log_metrics({
                "train_accuracy": train_report["accuracy"],
                "train_precision": train_report["1"]["precision"],
                "train_recall": train_report["1"]["recall"],
                "train_f1-score": train_report["1"]["f1-score"],
                "test_accuracy": test_report["accuracy"],
                "test_precision": test_report["1"]["precision"],
                "test_recall": test_report["1"]["recall"],
                "test_f1-score": test_report["1"]["f1-score"]
            })

            # Keep the model with the best cross-validated F1 (not the test set, to avoid leakage)
            if grid_search.best_score_ > best_overall_score:
                best_overall_score = grid_search.best_score_
                best_overall_name = name
                best_overall_model = best_model

    print(f"\nBest model: {best_overall_name} (CV F1 = {best_overall_score:.4f})")
    mlflow.log_param("best_model", best_overall_name)
    mlflow.log_metric("best_model_cv_f1", best_overall_score)

    # Save the best model so the pipeline can commit it to the repo, and log
    # it as an MLflow artifact for traceability
    os.makedirs("engine_maintenance_project/deployment", exist_ok=True)
    model_path = "engine_maintenance_project/deployment/best_engine_maintenance_model_v1.joblib"
    joblib.dump(best_overall_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
