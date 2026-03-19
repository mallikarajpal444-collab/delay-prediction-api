import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("shipment_master_dataset.csv")

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

print("Columns:", df.columns)

FEATURES = [
    "distance_km",
    "route_congestion_score",
    "weather_risk_score",
    "carrier_avg_delay_minutes",
    "warehouse_congestion_score"
]

TARGET = "delay_minutes"

# 🔥 Fix decimal format issue
df[TARGET] = df[TARGET].astype(str).str.replace(",", ".").astype(float)

# 🔥 Ensure all features are numeric
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing values
df = df.dropna(subset=FEATURES + [TARGET])

# Split
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "delay_model_clean.joblib")

print("✅ Model trained successfully")