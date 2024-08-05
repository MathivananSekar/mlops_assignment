import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(n_estimators, max_depth):
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(x_train, y_train)

        accuracy = model.score(x_test, y_test)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, 'model.joblib')

        print(f"Model trained with accuracy: {accuracy}")


if __name__ == "__main__":
    # train_model(n_estimators=10, max_depth=5)
    # Run with different parameters
    # train_model(n_estimators=20, max_depth=10)
    train_model(n_estimators=30, max_depth=15)

