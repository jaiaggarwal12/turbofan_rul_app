import os
import numpy as np
import pandas as pd
import pickle           # <-- ADDED
from sklearn.preprocessing import MinMaxScaler
from src.data_loading import load_fd001
from src.config import SEQ_LEN, FEATURE_COLS, MAX_RUL, DATA_RAW_DIR, DATA_PROCESSED_DIR


# ---------------------------------------------------------
# Compute RUL for training engines
# ---------------------------------------------------------
def compute_rul_for_training(train_df):
    train_df = train_df.copy()

    # Max cycle for each engine = failure point
    max_cycles = train_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]

    # Merge with train DF
    train_df = train_df.merge(max_cycles, on="engine_id")

    # RUL formula
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]

    # Cap RUL to MAX_RUL
    train_df["RUL"] = train_df["RUL"].clip(upper=MAX_RUL)

    # Drop helper col
    train_df = train_df.drop(columns=["max_cycle"])

    return train_df


# ---------------------------------------------------------
# Scale train + test using MinMaxScaler on train only
# ---------------------------------------------------------
def scale_features(train_df, test_df):
    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLS])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    train_scaled[FEATURE_COLS] = scaler.transform(train_df[FEATURE_COLS])
    test_scaled[FEATURE_COLS] = scaler.transform(test_df[FEATURE_COLS])

    return scaler, train_scaled, test_scaled


# ---------------------------------------------------------
# Create sliding windows for TRAINING/VALIDATION
# ---------------------------------------------------------
def create_train_sequences(train_df):
    X, y = [], []

    for eng_id, df_engine in train_df.groupby("engine_id"):
        df_engine = df_engine.sort_values("cycle")

        feature_array = df_engine[FEATURE_COLS].values
        rul_array = df_engine["RUL"].values

        if len(df_engine) < SEQ_LEN:
            continue

        for i in range(len(df_engine) - SEQ_LEN):
            X.append(feature_array[i:i + SEQ_LEN])
            y.append(rul_array[i + SEQ_LEN])

    return np.array(X), np.array(y)


# ---------------------------------------------------------
# Create FINAL TEST sequences (one per engine)
# ---------------------------------------------------------
def create_test_sequences(test_df):
    X_test = []
    engine_ids = sorted(test_df["engine_id"].unique())

    for eng_id in engine_ids:
        df_engine = test_df[test_df["engine_id"] == eng_id].sort_values("cycle")
        feats = df_engine[FEATURE_COLS].values

        if len(feats) < SEQ_LEN:
            pad_rows = SEQ_LEN - len(feats)
            padded = np.vstack([np.zeros((pad_rows, len(FEATURE_COLS))), feats])
            window = padded
        else:
            window = feats[-SEQ_LEN:]

        X_test.append(window)

    return np.array(X_test)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():

    print("📥 Loading FD001 raw data...")
    train_df, test_df, rul_df = load_fd001(DATA_RAW_DIR)

    print("🔧 Computing RUL for training set...")
    train_df = compute_rul_for_training(train_df)

    print("📊 Scaling features...")
    scaler, train_scaled, test_scaled = scale_features(train_df, test_df)

    print("📐 Creating training/validation sequences...")
    X_train_all, y_train_all = create_train_sequences(train_scaled)

    # Split train into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=42
    )

    print(f"✔ X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"✔ X_val:   {X_val.shape}, y_val:   {y_val.shape}")

    print("✨ Creating TEST sequences...")
    X_test = create_test_sequences(test_scaled)

    # True test RUL from NASA (vector of length N_engines)
    y_test = rul_df["RUL"].values[: len(X_test)]

    print(f"✔ X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Ensure processed folder exists
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    print("💾 Saving processed arrays...")
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(DATA_PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_PROCESSED_DIR, "y_test.npy"), y_test)

    # ---------------------------------------------------------
    # NEW: Save the MinMaxScaler to scaler.pkl
    # ---------------------------------------------------------
    print("💾 Saving MinMaxScaler (scaler.pkl)...")
    with open(os.path.join(DATA_PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("🎉 Preprocessing complete!")
    print("Files saved in data/processed/")


if __name__ == "__main__":
    main()
