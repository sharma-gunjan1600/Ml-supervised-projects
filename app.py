
from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

# Load the saved pipeline
with open('final_pipeline.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', result=None, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        print("Form Data:", data)  # Debugging

        # Convert numeric fields to float
        for key in data:
            try:
                data[key] = float(data[key])
            except ValueError:
                pass  # Keep as string for categorical

        # Compute log columns
        if 'matureseeddw' in data and data['matureseeddw'] > 0:
            data['logmatureseeddw'] = np.log(data['matureseeddw'] + 1)
        if 'immatureseeddw' in data and data['immatureseeddw'] > 0:
            data['logimmatureseeddw'] = np.log(data['immatureseeddw'] + 1)
        if 'matureseeddw' in data and 'immatureseeddw' in data:
            total_seed_dw = data['matureseeddw'] + data['immatureseeddw']
            data['logtotalseeddw'] = np.log(total_seed_dw + 1)

        # Ensure all expected columns exist
        expected_cols = model_pipeline.named_steps['preprocessor'].feature_names_in_
        for col in expected_cols:
            if col not in data:
                data[col] = 'Unknown' if col in ['Species', 'site', 'plot', 'trt', 'Treatment'] else 0

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Prediction
        prediction = model_pipeline.predict(input_df)
        print("Prediction:", prediction)  # Debugging

        return render_template('index.html',
                               result=f"Predicted Total Seed Dry Weight: {prediction[0]:.4f}",
                               form_data=data)
    except Exception as e:
        print("Error:", e)
        return render_template('index.html', result=f"Error: {str(e)}", form_data=data)

if __name__ == '__main__':
    app.run(debug=True)
