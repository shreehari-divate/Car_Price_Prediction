import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, render_template, request, url_for
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            logging.info("Received POST request for prediction.")

            # Capture and log form data
            form_data = request.form.to_dict()
            logging.info(f"Form Data: {form_data}")

            car_name = form_data.get("name")
            data = CustomData(
                year=form_data.get("year"),
                kmdriven=form_data.get("kmdriven"),
                fuel=form_data.get("fuel"),
                engine=form_data.get('engine'),
                mileage=form_data.get("mileage"),
                power=form_data.get("power"),
                transmission=form_data.get("transmission"),
                seller=form_data.get("seller"),
                owner=form_data.get("owner")
            )

            pred_df = data.get_data_as_df()
            logging.info(f"Constructed DataFrame: {pred_df}")

            predict_pipeline = PredictPipeline()

            # Log model prediction process
            logging.info("Starting prediction.")
            results = predict_pipeline.predict(pred_df)[0]
            logging.info(f"Prediction Result: {results}")

            res_msg = f"Your car {car_name}'s approximate price is Rs. {results:.2f}"
            logging.info(f"Response Message: {res_msg}")

            return render_template('index.html', results=res_msg)
        
        except Exception as e:
            logging.error(f"Error occurred during prediction: {e}", exc_info=True)
            return render_template('index.html', results="An error occurred during prediction.")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()
