import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True, help='Target directory of the dataset')
    parser.add_argument("--data_dir", type=str, required=True, help='Directory of the unsorted dataset')
    parser.add_argument("--log_path", type=str, required=True, help='Path of the csv log file of the dataset')
    parser.add_argument("--test_size", type=int, default=200, help='test size')
    parser.add_argument("--val_size", type=int, default=200, help='val size')
    args = parser.parse_args()

log_path = args.log_path
rir_dir = args.data_dir
output_dir = args.target_dir

target_test_size = args.test_size
target_val_size = args.val_size
random_seed = 42

os.makedirs(output_dir, exist_ok=True)

rirs = pd.read_csv(log_path)

rirs["rt60_bin"] = pd.qcut(rirs["rt60"], q=10, labels=False)

df_remain, df_test = train_test_split(
    rirs,
    test_size=target_test_size,
    stratify=rirs["rt60_bin"],
    random_state=random_seed
)

df_remain["rt60_bin"] = pd.qcut(df_remain["rt60"], q=10, labels=False)
df_train, df_val = train_test_split(
    df_remain,
    test_size=target_val_size,
    stratify=df_remain["rt60_bin"],
    random_state=random_seed
)

splits = {
    "test": df_test,
    "val": df_val,
    "train": df_train
}

for split_name, split_df in splits.items():
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    csv_output_path = os.path.join(output_dir, f"{split_name}.csv")
    split_df.drop(columns=["rt60_bin"]).to_csv(csv_output_path, index=False)

for split_name, split in splits.items():
    for filename, plugin in zip(split["RIR"], split["Plugin Name"]):
        src_path = os.path.join(rir_dir, plugin, filename)
        dst_path = os.path.join(output_dir, split_name, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Datei nicht gefunden: {src_path}")
    
for name, subset in splits.items():
    mean_rt60 = subset["rt60"].mean()
    print(f"{name.upper():5s}: {len(subset)} files, avg rt60: {mean_rt60:.3f}")

