from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("lung_cancer_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [int(request.form[key]) for key in request.form]
    prediction = model.predict([features])[0]
    result = "High Risk of Lung Cancer!!!" if prediction == 1 else "Low Risk of Lung Cancer!!!"
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
