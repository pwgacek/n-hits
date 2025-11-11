import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def std_scaler_np(x):
    mean = np.mean(x)
    std = np.std(x)  # ddof=0 (population)
    return (x - mean) / std

def transform(file_path):
    # === Load the dataset ===
    df = pd.read_csv(file_path)

    feature_order = [col for col in df.columns if col != "date"]

    # === Reshape from wide to long ===
    df_long = df.melt(id_vars="date", var_name="unique_id", value_name="y")

    # === Prepare folder for output ===
    folder_name = os.path.splitext(file_path)[0]  # "weather"
    output_dir = os.path.join(folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # === Apply StandardScaler per unique_id ===
    scalers = {}
    df_long["y_scaled"] = 0.0  # placeholder column

    for var in feature_order:  # maintain original order
        mask = df_long["unique_id"] == var
        scaler = StandardScaler()
        df_long.loc[mask, "y_scaled"] = scaler.fit_transform(df_long.loc[mask, ["y"]]).ravel()
        scalers[var] = scaler  # optional: keep for inverse transform

    # === Replace y with scaled values ===
    df_long["y"] = df_long["y_scaled"]
    df_long.drop(columns="y_scaled", inplace=True)

    # === Rename columns to match N-BEATSx format ===
    df_long.rename(columns={"date": "ds"}, inplace=True)

    # === Sort by ds first, then by original feature order ===
    df_long["unique_id"] = pd.Categorical(df_long["unique_id"], categories=feature_order, ordered=True)
    df_long = df_long.sort_values(by=["ds", "unique_id"]).reset_index(drop=True)

    # === Save result ===
    output_path = os.path.join(output_dir, "df_y.csv")
    df_long.to_csv(output_path, index=False)