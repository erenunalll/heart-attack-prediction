import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("dataset/processed/processed_Medicaldataset.csv")

y = df["Result"]
X = df.drop(columns="Result")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# scaling X
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# loading models
decision_tree_model = joblib.load("models/decision_tree_model.joblib")
random_forest_model = joblib.load("models/random_forest_model.joblib")

logistic_regression_model = joblib.load("models/logistic_regression_model.joblib")
knn_model = joblib.load("models/knn_model.joblib")
support_vector_model = joblib.load("models/support_vector_model.joblib")

predictions = {
    "Decision Tree": decision_tree_model.predict(X_test),
    "Random Forest": random_forest_model.predict(X_test),
    "Logistic Regression": logistic_regression_model.predict(X_test_scaled),
    "KNN": knn_model.predict(X_test_scaled),
    "SVM": support_vector_model.predict(X_test_scaled)
}

for name, y_pred in predictions.items():
    print(f"======== {name} ========")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()

# Random Forest and Decision Tree classifiers are better
print("="*50)
print("="*50)
print("="*50)
print()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=2026
)

predictions = {
    "Decision Tree": decision_tree_model.predict(X_test),
    "Random Forest": random_forest_model.predict(X_test)
}

for name, y_pred in predictions.items():
    print(f"======== {name} ========")
    print(f"Prediction:\n {y_pred}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()