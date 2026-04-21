import os
import pandas as pd

input_dir = "dataset"
output_dir = "dataset_sample"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_list = [
    "pai_machine_spec.csv",
    "pai_machine_metric.csv",
    "pai_task_table.csv"
]

for file_name in file_list:
    header_path = os.path.join(input_dir, file_name.replace(".csv", ".header"))
    with open(header_path, "r", encoding="utf-8") as f:
        columns = f.read().strip().split(",")

    df = pd.read_csv(os.path.join(input_dir, file_name), header=None)
    df.columns = columns

    df.to_csv(os.path.join(output_dir, file_name), index=False, encoding="utf-8")