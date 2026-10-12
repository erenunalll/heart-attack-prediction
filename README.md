# Prediction of Heart Attack Probability
This project aims to predict the probability of a heart attack using supervised machine learning methods and patient health data.

## Dataset

The dataset used in this project contains the following features:
- Age
- Gender
- Heart Rate
- Systolic blood pressure
- Diastolic blood pressure
- Blood sugar
- CK-MB 
- Troponin
- Result

The dataset is obtained from Kaggle:

[Heart Attack Dataset - Kaggle](https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid)


## Project Structure
```
heart-attack-prediction/
├── .gitignore
├── app/
│   └── app.py
├── dataset/
│   ├── processed/
│   │   └── processed_Medicaldataset.csv
│   └── raw/
│       └── Medicaldataset.csv
├── models/
├── notebooks/
│   └── eda.ipynb
├── plots/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── eval.py
    ├── predict.py
    ├── preprocess.py
    └── train.py
```

## Methodology
The project follows these main steps:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training
4. Model Evaluation and Comparison
5. Heart Attack Risk Prediction

## Installation
Clone the repository:
```bash
git clone <repository-url>
cd heart-attack-prediction
```

Install the required dependencies
```bash
pip install -r requirements.txt
```

## Disclaimer

This project is intended for educational purposes only. Its predictions should not be used as a substitute for professional medical advice or diagnosis.