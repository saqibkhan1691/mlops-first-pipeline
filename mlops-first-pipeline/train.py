from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import mlflow
import mlflow.sklearn

#Client API lets us manage registered models and their aliases
from mlflow import MlflowClient

# Configure MLFLow Tracking
# MLFlow server stores experiment metadata and model records
mlflow.set_tracking_uri("http://127.0.0.1:6001")

# Group related training runs under one experiment
mlflow.set_experiment("Iris Classification")


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_estimators=10
random_state=42


# Model settings we want to track in MLFlow
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.2
}

#Start a tracked training run
with mlflow.start_run() as run:
    # Record the training configuration before model fitting
    mlflow.log_params(model_parameters)

    #Build classifier using the selected parameters
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
        )

    #Fit the model on training data
    model.fit(X_train, y_train)

    #Generate prediction using test data
    y_pred = model.predict(X_test)

    #Calculate metrics that help us evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # Model Quality Gate
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected: test accuracy {accuracy:.2f} is below {MIN_TEST_ACCURACY}"
        )

    print("Model Quality Gate Passed. Model is eligible for registration")

    #Log the evaluated results so that different runs can be compared
    mlflow.log_metrics({
        "test_Accuracy": accuracy,
        "test_Precision": precision,
        "test_Recall": recall
    })

    print("MLFlow Run ID:", run.info.run_id)
    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)

    # save the trained model as resuable artifact
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "iris_model.joblib")
    print(f"Model saved to {model_dir / 'iris_model.joblib'}")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris_model",
        registered_model_name="iris_classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )
    

    # Assign the "champion" alias to the newly registered model
    client = MlflowClient()

    # Identify the curent training run
    current_run_id = mlflow.active_run().info.run_id

    # Retrieve all version registered under this model name
    registered_versions = client.search_model_versions(
        "name='iris_classifier'"
    )

    # Find the version created by the training run
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None
    )

    if current_model_version is None:
        raise RuntimeError(
            "Model was logged but its registered version could not be found"
        )

    # Assign the alias only after model has passed quality gate
    client.set_registered_model_alias(
        name="iris_classifier",
        alias="champion",
        version=current_model_version.version
    )

    print(
        f"Champion alias assigned to model version"
        f"{current_model_version.version}"
    )

    print("Registered Model: iris_classifier")