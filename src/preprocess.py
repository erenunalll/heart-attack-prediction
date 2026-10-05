import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("dataset/raw/Medicaldataset.csv")

    # numerize categorical features
    result_map = {
        "negative":0,
        "positive":1
    }
    df["Result"] = df["Result"].map(result_map)

    heart_rate_mask = (df["Heart rate"] < 200) | (df["Heart rate"] > 30)
    df = df[heart_rate_mask]

    df.to_csv("dataset/processed/processed_Medicaldataset.csv", index=False)
   

if __name__ == "__main__":
    main()
