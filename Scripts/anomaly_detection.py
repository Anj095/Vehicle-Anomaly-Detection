import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


# LOAD DATA CORRECTLY (WORKS ANYWHERE)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to CSV
file_path = os.path.join(BASE_DIR, "mock_vehicle_data.csv")

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Columns detected:", df.columns.tolist())

# Clean column names (remove hidden spaces if any)
df.columns = df.columns.str.strip()

# Convert timestamp column
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Drop rows with invalid timestamps
df = df.dropna(subset=["timestamp"])

 
# SELECT FEATURES (MATCH YOUR DATA)
 

features = df[["speed", "fuel_level", "engine_temp"]]

 
# ANOMALY DETECTION
 

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

df["anomaly"] = model.fit_predict(features)

# Convert model output (-1 anomaly, 1 normal)
df["anomaly"] = np.where(df["anomaly"] == -1, 1, 0)

print("Total anomalies detected:", df["anomaly"].sum())

 
# SAVE RESULTS
 

output_path = os.path.join(BASE_DIR, "vehicle_anomalies_detected.csv")
df.to_csv(output_path, index=False)

print("Results saved to:", output_path)

  
# VISUALIZATION (Seaborn)
 

sns.set_style("whitegrid")

plt.figure()
sns.scatterplot(
    data=df,
    x="engine_temp",
    y="speed",
    hue="anomaly"
)
plt.title("Engine Temp vs Speed (Anomalies Highlighted)")
plt.show()
