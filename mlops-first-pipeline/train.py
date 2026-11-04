from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import precision_score
import recall_score

import mlflow
import mlflow.sklearn


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Configure MLFlow Tracking
# MLFLow server stores experiment metadata and model records
mlflow.set_tracking_uri("http://0.0.0.0:6001")



n_estimators = 100
random_state = 42

model_parameters = (
    "model_type", "RandomForestClassifier",
    "n_estimators", n_estimators,
    "random_state", random_state,
    "test_size", 0.2
)

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
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

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

    print("Model logged to MLFlow and registered as 'iris_classifier'")


# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model accuracy: {accuracy}")

# model_dir = Path("models")
# model_dir.mkdir(parents=True, exist_ok=True)
# joblib.dump(model, model_dir / "iris_model.joblib")
# print(f"Model saved to {model_dir / 'iris_model.joblib'}")