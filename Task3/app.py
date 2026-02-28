from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    attendance = float(request.form["attendance"])

    data = np.array([[hours, attendance]])
    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    result = "PASS ✅" if prediction == 1 else "FAIL ❌"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)