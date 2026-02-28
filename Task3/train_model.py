import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Sample dataset
data = {
    "hours_study": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [50,55,60,65,70,75,80,85,90,95],
    "pass": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["hours_study", "attendance"]]
y = df["pass"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Save model + scaler
joblib.dump(model, "student_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained and saved!")