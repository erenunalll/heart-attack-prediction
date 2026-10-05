import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler


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

# tree based classifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# scaled classifiers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_scaled, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_scaled, y_train)

support_vector_model = SVC(random_state=42)
support_vector_model.fit(X_scaled, y_train)

joblib.dump(decision_tree_model ,"models/decision_tree_model.joblib")
joblib.dump(random_forest_model, "models/random_forest_model.joblib")

joblib.dump(logistic_regression_model, "models/logistic_regression_model.joblib")
joblib.dump(knn_model, "models/knn_model.joblib")
joblib.dump(support_vector_model, "models/support_vector_model.joblib")