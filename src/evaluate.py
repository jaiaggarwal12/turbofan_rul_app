import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

from src.model import AttentionLayer
from src.config import DATA_PROCESSED_DIR, MODEL_SAVE_PATH


def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs True RUL")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_error_histogram(errors, save_path):
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Prediction Error (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("RUL Prediction Error Distribution")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    print("📥 Loading processed test dataset...")
    X_test = np.load(os.path.join(DATA_PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PROCESSED_DIR, "y_test.npy"))

    print(f"✔ X_test: {X_test.shape}, y_test: {y_test.shape}")

    print("📥 Loading best model...")
    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH,
        custom_objects={"AttentionLayer": AttentionLayer}
    )

    print("🔮 Predicting on test engines...")
    y_pred = model.predict(X_test).flatten()

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 FINAL TEST METRICS (FD001):")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

    os.makedirs("reports/figures", exist_ok=True)

    plot_predictions(y_test, y_pred, "reports/figures/pred_vs_true.png")
    plot_error_histogram(y_test - y_pred, "reports/figures/error_histogram.png")

    print("📁 Plots saved in reports/figures/")
    print("🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
