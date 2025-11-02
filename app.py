from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('loan_model.pkl')  # Ensure this file exists in the same folder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            int(request.form['gender']),
            int(request.form['married']),
            int(request.form['education']),
            int(request.form['self_employed']),
            float(request.form['applicantincome']),
            float(request.form['coapplicantincome']),
            float(request.form['loanamount']),
            float(request.form['loan_amount_term']),
            float(request.form['credit_history']),
            int(request.form['property_area'])
        ]

        prediction = model.predict([data])
        result = 'Approved ✅' if prediction[0] == 1 else 'Rejected ❌'
        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
