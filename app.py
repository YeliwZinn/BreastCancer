from flask import Flask, render_template, request
from implementation import random_forest_train, random_forest_predict
import numpy as np
import time
from sklearn.preprocessing import StandardScaler  # Import for data scaling (optional)

# Error handling imports (consider adding these)
from werkzeug.exceptions import BadRequest  # For handling bad request format

app = Flask(__name__)
app.url_map.strict_slashes = False

# Global model variable (consider loading the model at runtime)
clf = None  # Initialize as None to handle potential training errors

@app.route('/')
def index():
    return render_template('home.html')  # Assuming home.html exists


@app.route('/predict', methods=['POST'])
def predict():
    data_points = []
    data = []

    try:
        # Extract features from the request (improved error handling)
        for i in range(1, 31):
            value = request.form.get('value{}'.format(i))
            if value is None:
                raise BadRequest("Missing data point: value{}".format(i))
            data.append(float(value))

        # Convert data to NumPy array and reshape
        data_np = np.asarray(data, dtype=float)
        data_np = data_np.reshape(1, -1)

        # Check if model is trained
        if clf is None:
            # Handle the case where the model hasn't been trained yet
            return render_template('error.html', message="Model is not yet trained. Please wait.")

        # Make prediction using the trained model
        out, acc, prediction_time = random_forest_predict(clf, data_np)

        # Interpret prediction and accuracy
        if out == 1:
            output = 'Malignant'
        else:
            output = 'Benign'

        acc_x, acc_y = acc[0]
        accuracy = max(acc_x, acc_y)  # Use the higher accuracy value

        return render_template('result.html', output=output, accuracy=accuracy, time=prediction_time)

    except (ValueError, BadRequest) as e:
        # Handle potential errors during data conversion or missing values
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    # Train the model (consider moving this to a separate script)
    try:
        clf = random_forest_train()  # Assuming random_forest_train trains the model
    except Exception as e:
        print(f"Error during model training: {e}")
        clf = None  # Set clf to None to indicate training failure

    # Run the Flask app in debug mode
    app.run(debug=True)
