import pandas as pd
df = pd.read_parquet("data/training_dataset.parquet")
print("Columnas en training_dataset.parquet:", df.columns.tolist()[:15])
print("Variables objetivo encontradas:", [c for c in df.columns if "target" in c or "win" in c or "runs" in c])
