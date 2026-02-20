from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('job_placement_model.pkl')

# compatibility fix: older pickled trees may not have the
# `monotonic_cst` attribute which newer sklearn versions check for
# during prediction.  Ensure every estimator has the attribute.
if hasattr(model, 'estimators_'):
    for est in model.estimators_:
        if not hasattr(est, 'monotonic_cst'):
            est.monotonic_cst = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = request.form['gender']
    stream = request.form['stream']
    internships = int(request.form['internships'])
    cgpa = float(request.form['cgpa'])
    history_of_backlogs = int(request.form['history_of_backlogs'])

    gender_encoded = 1 if gender == 'Male' else 0
    stream_mapping = {
        'Civil': 0,
        'Computer Science': 1,
        'Electrical': 2,
        'Electronics And Communication': 3,
        'Information Technology': 4,
        'Mechanical': 5
    }
    stream_encoded = stream_mapping.get(stream, -1)

    if stream_encoded == -1:
        return render_template('index.html', error="Invalid stream selected.")

    # construct a DataFrame with the same feature names used during training
    input_df = pd.DataFrame(
        [[age, gender_encoded, stream_encoded, internships, cgpa, history_of_backlogs]],
        columns=['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'HistoryOfBacklogs']
    )

    prediction = model.predict(input_df)

    result = "The student is likely to be placed." if prediction[0] == 1 else "The student is unlikely to be placed."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
