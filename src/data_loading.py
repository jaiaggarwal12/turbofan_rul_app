import os
import pandas as pd

def load_fd001(data_raw_dir):
    col_names = [
        "engine_id", "cycle",
        "op_setting_1", "op_setting_2", "op_setting_3",
    ] + [f"s{i}" for i in range(1, 22)]

    train_df = pd.read_csv(
        os.path.join(data_raw_dir, "train_FD001.txt"),
        sep=r"\s+", header=None, names=col_names
    )
    test_df = pd.read_csv(
        os.path.join(data_raw_dir, "test_FD001.txt"),
        sep=r"\s+", header=None, names=col_names
    )
    rul_df = pd.read_csv(
        os.path.join(data_raw_dir, "RUL_FD001.txt"),
        sep=r"\s+", header=None, names=["RUL"]
    )

    return train_df, test_df, rul_df
