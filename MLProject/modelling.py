import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Set experiment 
mlflow.set_experiment("ci-training")

# 2. Load dataset
data = pd.read_csv("heart_preprocessed.csv")

X = data.drop("target", axis=1)
y = data["target"]

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_metric("accuracy", acc)


mlflow.sklearn.log_model(model, "model")

print(f"Training selesai, accuracy: {acc}")
