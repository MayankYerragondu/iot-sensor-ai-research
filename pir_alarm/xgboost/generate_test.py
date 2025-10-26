import pandas as pd
import numpy as np

# -----------------------------
# Synthetic PIR Alarm Dataset
# -----------------------------
n = 100_000
np.random.seed(42)

# Base features
hours = np.random.randint(0, 24, n)
minutes = np.random.randint(0, 60, n)
days = np.random.randint(0, 7, n)

# Simulate realistic time differences (seconds between triggers)
# Normal events: moderate gaps (30–300s)
# False alarms: very short (<10s) or very long (>600s)
time_diff = np.random.choice(
    np.concatenate([
        np.random.uniform(30, 300, int(n * 0.9)),   # normal
        np.random.uniform(0, 10, int(n * 0.05)),    # too frequent (false)
        np.random.uniform(600, 1200, int(n * 0.05)) # too sparse (false)
    ]),
    n,
    replace=False
)

# Create basic frame
df = pd.DataFrame({
    "hour": hours,
    "minute": minutes,
    "day_of_week": days,
    "time_diff": time_diff
})

# Label generation rule (for supervision)
# Higher chance of false alarm at night (low activity) or extreme time_diff
false_alarm_prob = (
    (df["hour"].between(0, 5)) * 0.6 +          # more false alarms midnight–5 AM
    (df["time_diff"] < 10) * 0.7 +              # bursts
    (df["time_diff"] > 600) * 0.5 +             # long inactivity
    np.random.rand(n) * 0.1                     # noise
)

df["label"] = (false_alarm_prob > 0.8).astype(int)

# Randomize order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("piralarm_xgboost_training.csv", index=False)

print(df.head())
print("\n✅ Saved 100,000-row synthetic dataset → piralarm_xgboost_training.csv")
print(df["label"].value_counts(normalize=True))
