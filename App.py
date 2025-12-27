from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("loan_pipeline.pkl")

@app.route("/")
def home():
    return "Loan Approval Model is Live 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return jsonify({
        "loan_approved": int(prediction),
        "approval_probability": round(float(prob), 3)
    })

if __name__ == "__main__":
    app.run()
