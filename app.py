import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)

# Load models
model1 = pickle.load(open('xgboost_model.pkl', 'rb'))
# Load other models as needed

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""

    if request.method == 'POST':
        try:
            # Get data from the uploaded CSV file
            csv_file = request.files['csv_file']
            if csv_file:
                # Assuming you want to make predictions on the data in the CSV file
                data_df = pd.read_csv(csv_file)
                
                # Assuming your model expects an array-like input
                result = model1.predict(data_df.values.reshape(1, -1))
                
                # Depending on the nature of your prediction, you might need to convert the result to a meaningful text
                prediction_text = f"Predicted result: {result[0]}"
                # Include predictions from other models as needed

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
