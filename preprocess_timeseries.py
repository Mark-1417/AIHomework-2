import os
import pandas as pd

sample_dir = "dataset_sample"

spec = pd.read_csv(os.path.join(sample_dir, "pai_machine_spec.csv"))
metric = pd.read_csv(os.path.join(sample_dir, "pai_machine_metric.csv"))

merged = pd.merge(metric, spec, on="machine", how="inner")

use_cols = [
    "machine", "start_time",
    "machine_cpu", "machine_gpu",
    "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
    "machine_load_1", "machine_net_receive",
    "cap_cpu", "cap_mem", "cap_gpu"
]
df_final = merged[use_cols]

num_cols = [c for c in use_cols if c not in ["machine"]]
for col in num_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")
df_final = df_final.dropna()

for col in ["machine_cpu", "machine_gpu", "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr"]:
    df_final[col] = df_final[col].clip(0, 100)

df_final = df_final.sort_values(["machine", "start_time"]).reset_index(drop=True)
df_final.to_csv(os.path.join(sample_dir, "merged_final_data.csv"), index=False)