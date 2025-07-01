import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('flightdata.csv')

# Drop rows with missing values in important columns
important_columns = ['CRS_DEP_TIME', 'DAY_OF_WEEK', 'FL_NUM', 'DISTANCE',
                     'UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'ARR_DEL15']
df.dropna(subset=important_columns, inplace=True)

# Create target variable (1 = Delayed, 0 = On Time)
df['IsDelayed'] = df['ARR_DEL15'].astype(int)

# Extract hour from CRS_DEP_TIME
df['DEP_HOUR'] = df['CRS_DEP_TIME'].astype(str).str.zfill(4).str[:2].astype(int)

# Encode categorical variables
encoders = {}
for col in ['UNIQUE_CARRIER', 'ORIGIN', 'DEST']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col.lower()] = le

# Features and target
features = ['DEP_HOUR', 'DAY_OF_WEEK', 'FL_NUM', 'DISTANCE',
            'UNIQUE_CARRIER', 'ORIGIN', 'DEST']
X = df[features]
y = df['IsDelayed']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("✅ Model and encoders saved successfully.")
