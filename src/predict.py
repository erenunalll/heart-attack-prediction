import joblib
import numpy as np


def load_model():
    return joblib.load("models/decision_tree_model.joblib")


def predict(model, patient_data):
    patient_data = np.array(patient_data).reshape(1, -1)
    model = load_model()

    prediction = model.predict(patient_data)[0]
    probabilities = model.predict_proba(patient_data)[0]

    return {
        "prediction": int(prediction),
        "probability": float(probabilities[1] * 100)
    }
    

if __name__ == "__main__":
    pass