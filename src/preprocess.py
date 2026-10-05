import pandas as pd
import numpy as np


def save_dataset(df: pd.DataFrame) -> None:
    df.to_csv("dataset/processed/processed_Medicaldataset.csv")

    return


def main():
    df = pd.read_csv("dataset/raw/Medicaldataset.csv")

    # numerize categorical features
    result_map = {
        "negative":0,
        "positive":1
    }
    df["Result"] = df["Result"].map(result_map)

    heart_rate_mask = (df["Heart Rate"] < 200) | (df["Heart Rate"] > 30)
    df = df[heart_rate_mask]

    print(df.head())


if __name__ == "__main__":
    main()
