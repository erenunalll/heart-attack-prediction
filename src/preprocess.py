import pandas as pd
import numpy as np


def save_dataset(df: pd.DataFrame) -> None:
    df.to_csv("dataset/processed/processed_Medicaldataset.csv")

    return


def main():
    df = pd.read_csv("dataset/raw/Medicaldataset.csv")

    result_map = {
        "negative":0,
        "positive":1
    }
    df["Result"] = df["Result"].map(result_map)

    print(df.head())


if __name__ == "__main__":
    main()
