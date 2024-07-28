import logging
import sys
import os

from sklearn.preprocessing import OneHotEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from flask import Flask, render_template, request, url_for
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            car_name = request.form.get("name")
            year = request.form.get("year")
            km_driven = request.form.get("km_driven")
            engine = request.form.get('engine')
            mileage = request.form.get("mileage")
            max_power = request.form.get("max_power")
            fuel = request.form.get("fuel")
            transmission = request.form.get("transmission")
            seller_type = request.form.get("seller_type")
            owner = request.form.get("owner")

            logger.info("Received form data: %s, %s, %s, %s, %s, %s, %s, %s, %s, %s",
                        car_name, year, km_driven, engine, mileage, max_power, fuel, transmission, seller_type, owner)

            year = int(year)
            km_driven = int(km_driven)
            engine = int(engine)
            mileage = float(mileage)
            max_power = float(max_power)
            data = CustomData(
                year=year,
                kmdriven=km_driven,
                fuel=fuel,
                engine=engine,
                mileage=mileage,
                power=max_power,
                transmission=transmission,
                seller=seller_type,
                owner=owner
            )
            
            pred_df = data.get_data_as_df()
            logger.info("Dataframe columns: %s", pred_df.columns)
            logger.info("Dataframe: %s", pred_df)

            predict_pipeline = PredictPipeline()  
            results = predict_pipeline.predict(pred_df)[0]

            if not isinstance(results, (int, float)):
                raise ValueError("Prediction is not a number")
            
            res_msg = f"Your car {car_name}'s approximate price is Rs. {results:.2f}"
            return render_template('index.html', results=res_msg) 
        
        except Exception as e:
            logger.error("Error during prediction: %s", e)
            return render_template('index.html', error=str(e))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
