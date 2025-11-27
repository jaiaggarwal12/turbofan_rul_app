import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import DATA_PROCESSED_DIR, SEQ_LEN, MODEL_SAVE_PATH
from src.model import build_rul_model


def load_processed_data():
    X_train = np.load(os.path.join(DATA_PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PROCESSED_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_PROCESSED_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_PROCESSED_DIR, "y_val.npy"))

    return X_train, y_train, X_val, y_val


def plot_history(history, save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)

    # Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss (MAE)")
    plt.plot(history.history["val_loss"], label="Val Loss (MAE)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training & Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()


def main():
    print("📥 Loading processed data...")
    X_train, y_train, X_val, y_val = load_processed_data()
    print(f"✔ X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"✔ X_val:   {X_val.shape}, y_val:   {y_val.shape}")

    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]
    print(f"📐 seq_len = {seq_len}, num_features = {num_features}")

    print("🧠 Building model...")
    model = build_rul_model(seq_len=seq_len, num_features=num_features)
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_mae",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_mae",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
        mode="min",
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=15,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )

    EPOCHS = 100      # long training as you chose
    BATCH_SIZE = 64

    print("🚀 Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
        verbose=1,
    )

    print(f"💾 Best model saved to: {MODEL_SAVE_PATH}")

    print("📊 Plotting training curves...")
    plot_history(history)

    print("🎉 Training complete!")


if __name__ == "__main__":
    main()
