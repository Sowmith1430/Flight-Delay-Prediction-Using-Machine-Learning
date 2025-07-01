import pandas as pd

# Load your dataset
df = pd.read_csv("flightdata.csv")

# Filter for delayed flights (where delay >= 15 minutes)
delayed_flights = df[df['DEP_DEL15'] == 1]

# Display relevant info
print(delayed_flights[['FL_NUM', 'UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY']].head(10))
