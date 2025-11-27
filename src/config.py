SEQ_LEN = 50
MAX_RUL = 125

# Features used from dataset
FEATURE_COLS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
] + [f"s{i}" for i in range(1, 22)]

DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"

MODEL_SAVE_PATH = "saved_models/best_model.keras"
