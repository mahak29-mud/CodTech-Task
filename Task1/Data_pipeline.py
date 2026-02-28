# ETL Pipeline using Pandas & Scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


data = pd.read_csv("StudentPerformanceFactors.csv")

print("Original Data:")
print(data.head())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns


# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression())
])



model_pipeline.fit(X_train, y_train)



y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)



joblib.dump(model_pipeline, "etl_model_pipeline.pkl")

processed_data = preprocessor.fit_transform(X)
pd.DataFrame(processed_data).to_csv("processed_student_data.csv", index=False)

print("\nETL Process Completed Successfully!")